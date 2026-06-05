# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.repo_io import sha256_file
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hi_nerv.receiver_cache_quality import (
    HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA,
    HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA,
    HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA,
    build_hi_nerv_receiver_cache_segnet_argmax_probe,
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


def test_hi_nerv_receiver_cache_quality_uses_explicit_source_pair_indices(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")

    report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "quality",
        max_pairs=1,
        batch_pairs=1,
        pair_indices=(1,),
    )

    cache_dir = Path(report["candidate_cache_dir"])
    pair_indices = np.load(cache_dir / "pair_indices.npy")
    direct = report["direct_receiver_cache_report"]
    audit = json.loads(
        (cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json").read_text(
            encoding="utf-8"
        )
    )

    assert pair_indices.tolist() == [[2, 3]]
    assert direct["selected_pair_indices"] == [1]
    assert direct["pair_index_scope"] == "explicit_source_pair_indices"
    assert audit["direct_render"]["selected_pair_indices"] == [1]


def test_hi_nerv_receiver_cache_quality_requires_argmax_probe_for_reference_gate(
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
    assert report["quality_gate_passed"] is False
    assert report["segnet_argmax_probe"]["fit_gate_passed"] is False
    assert "hi_nerv_receiver_cache_segnet_argmax_probe_not_run" in report["blockers"]
    assert report["distortion_crux_probe"]["schema"] == (
        HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA
    )
    assert report["distortion_crux_probe"]["fit_gate_passed"] is True
    assert report["distortion_crux_probe"]["hard_pair_rows"][0]["pair_index"] == 0
    assert report["hard_pair_coverage"]["score_axis_hard_pair_coverage"] is False
    assert Path(report["segnet_argmax_probe_path"]).is_file()
    assert Path(report["distortion_crux_probe_path"]).is_file()
    assert report["score_claim"] is False
    assert Path(report["quality_gate_path"]).is_file()


def test_hi_nerv_receiver_cache_quality_passes_with_argmax_probe(
    tmp_path: Path,
) -> None:
    archive = _write_tiny_hiv1_archive(tmp_path / "archive.zip")
    reference_report = write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=tmp_path / "reference",
        max_pairs=1,
        batch_pairs=1,
    )

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        logits[:, 0, :, :] = 1.0
        return logits

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
        segnet_argmax_probe_logits_fn=fake_segnet_logits,
    )

    assert report["quality_gate_passed"] is True
    assert report["segnet_argmax_probe"]["scorer_backend"] == "injected_segnet_logits_fn"
    assert report["segnet_argmax_probe"]["fit_gate_passed"] is True
    assert report["segnet_argmax_probe"]["segnet_argmax_disagreement_rate"] == 0.0
    assert "candidate_segnet_argmax_disagreement_too_high" not in report["blockers"]
    assert Path(report["segnet_argmax_probe_path"]).is_file()


def test_hi_nerv_receiver_cache_segnet_argmax_probe_prices_real_flip_surface(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    cand = ref.copy()
    cand[0, 0, 0, 0] = 255.0
    np.save(reference / "segnet_last_rgb.npy", ref)
    np.save(candidate / "segnet_last_rgb.npy", cand)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        logits[:, 0, :, :] = 1.0
        logits[:, 1, :, :] = (x_nchw[:, 0, :, :] > 128.0).astype(np.float32) * 3.0
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=tmp_path / "upstream",
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=0.05,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["schema"] == HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA
    assert report["scorer_backend"] == "injected_segnet_logits_fn"
    assert report["total_pixels"] == 16
    assert report["mismatch_pixels"] == 1
    assert report["segnet_argmax_disagreement_rate"] == pytest.approx(1.0 / 16.0)
    assert report["fit_gate_passed"] is False
    assert "candidate_segnet_argmax_disagreement_too_high" in report["blockers"]


def test_hi_nerv_receiver_cache_segnet_argmax_probe_names_class_collapse(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_cache"
    reference = tmp_path / "reference_cache"
    candidate.mkdir()
    reference.mkdir()
    cand = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref[0, 0, :, :] = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(candidate / "segnet_last_rgb.npy", cand)
    np.save(reference / "segnet_last_rgb.npy", ref)

    def fake_segnet_logits(x_nchw: np.ndarray) -> np.ndarray:
        b, _c, h, w = x_nchw.shape
        logits = np.zeros((b, 5, h, w), dtype=np.float32)
        cls = (x_nchw[:, 0, :, :].astype(np.int64) % 5).reshape(b, h, w)
        for class_index in range(5):
            logits[:, class_index, :, :] = np.where(cls == class_index, 2.0, 0.0)
        return logits

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate,
        reference_cache_dir=reference,
        upstream_dir=None,
        sample_pairs=1,
        batch_frames=1,
        max_segnet_argmax_disagreement_for_fit_gate=0.05,
        segnet_logits_fn=fake_segnet_logits,
    )

    assert report["candidate_argmax_histogram"] == [16, 0, 0, 0, 0]
    assert report["reference_argmax_histogram"] == [4, 3, 3, 3, 3]
    assert report["candidate_occupied_class_fraction"] == pytest.approx(0.2)
    assert report["reference_occupied_class_fraction"] == pytest.approx(1.0)
    assert (
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
        in report["blockers"]
    )


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
