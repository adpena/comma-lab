# SPDX-License-Identifier: MIT
"""Tests for compact decoder codec sweep materialization."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from tac.substrates._shared.compact_decoder_codec_sweep import (
    adjudicate_compact_decoder_codec_sweep_with_replay,
    sweep_compact_decoder_codecs,
)
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import (
    pack_archive as pack_hi_nerv_archive,
)
from tac.substrates.hi_nerv.archive import (
    parse_archive as parse_hi_nerv_archive,
)
from tac.substrates.hi_nerv.archive import split_archive_sections
from tac.substrates.pact_nerv_selector_v4.archive import (
    pack_archive as pack_selector_v4_archive,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    parse_archive as parse_selector_v4_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    pack_archive as pack_vq_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    parse_archive as parse_vq_archive,
)


def _source_zip(path: Path, bin_bytes: bytes) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", bin_bytes)
    return path


def test_sweep_vq_archive_materializes_codec_variants_fail_closed(tmp_path: Path) -> None:
    torch.manual_seed(1)
    decoder = {"conv.weight": torch.randn(4, 3, 3, 3) * 0.02}
    codebook = torch.randn(8, 4) * 0.01
    indices = torch.tensor([0, 3, 5], dtype=torch.long)
    source_bin = pack_vq_archive(
        decoder,
        codebook,
        indices,
        {"embed_dim": 4},
        decoder_codec="int8_mixed",
        indices_codec="auto",
    )
    source = _source_zip(tmp_path / "source_vq.zip", source_bin)

    report = sweep_compact_decoder_codecs(
        source_archive_zip=source,
        output_dir=tmp_path / "sweep",
        decoder_codecs=("int8_mixed", "int8_scale_bundled", "portfolio_auto"),
        repo_root=Path.cwd(),
        run_receiver_proof=False,
    )

    rows = report["variant_rows"]
    assert report["family"] == "pact_nerv_vq"
    assert len(rows) == 3
    assert len(report["section_value_rows"]) == 3
    assert report["byte_price_plan"]["schema"] == (
        "compact_nerv_byte_price_controller.v1"
    )
    assert "delta_nonrate_score_missing" in report["byte_price_plan"]["blockers"]
    assert [row["archive_bytes"] for row in rows] == sorted(
        row["archive_bytes"] for row in rows
    )
    assert report["best_variant"]["archive_bytes"] == min(
        row["archive_bytes"] for row in rows
    )
    for row in rows:
        assert row["score_claim"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["charged_bits_changed"] is True
        assert row["score_affecting_payload_changed"] is True
        assert row["exact_axis_score_affecting_adjudication_required"] is True
        assert "receiver_proof_not_run" in row["blockers"]
        assert Path(row["archive_path"]).is_file()
        assert Path(row["bin_path"]).is_file()
        parse_vq_archive(Path(row["bin_path"]).read_bytes())


def test_sweep_selector_archive_materializes_codec_variants_fail_closed(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2)
    decoder = {"conv.weight": torch.randn(4, 3, 3, 3) * 0.02}
    latents = torch.randn(3, 4) * 0.01
    selector_bytes = b"\x00\x01\x01"
    source_bin = pack_selector_v4_archive(
        decoder,
        latents,
        selector_bytes,
        {"embed_dim": 4},
        palette_size=16,
        decoder_codec="int8_mixed",
    )
    source = _source_zip(tmp_path / "source_selector.zip", source_bin)

    report = sweep_compact_decoder_codecs(
        source_archive_zip=source,
        output_dir=tmp_path / "sweep_selector",
        decoder_codecs=("int8_mixed", "int8_scale_bundled"),
        repo_root=Path.cwd(),
        run_receiver_proof=False,
    )

    rows = report["variant_rows"]
    assert report["family"] == "pact_nerv_selector_v4"
    assert len(rows) == 2
    assert report["byte_price_plan"]["input_row_count"] == 2
    assert report["section_value_rows"][0]["scope"] == (
        "compact_decoder_codec_replacement"
    )
    for row in rows:
        assert row["promotion_eligible"] is False
        assert row["charged_bits_changed"] is True
        assert "receiver_proof_not_run" in row["blockers"]
        assert Path(row["archive_path"]).is_file()
        parse_selector_v4_archive(Path(row["bin_path"]).read_bytes())


def test_sweep_hi_nerv_archive_materializes_codec_variants_and_preserves_latents(
    tmp_path: Path,
) -> None:
    torch.manual_seed(3)
    cfg = HinervConfig(
        latent_dim_coarse=4,
        latent_dim_mid=5,
        latent_dim_fine=6,
        embed_dim=12,
        initial_grid_h=2,
        initial_grid_w=3,
        decoder_channels=(10, 8),
        num_upsample_blocks=2,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=3,
        output_height=8,
        output_width=12,
    )
    model = HinervSubstrate(cfg)
    sd = model.state_dict()
    decoder = {
        k: v.clone()
        for k, v in sd.items()
        if k not in ("latents_coarse", "latents_mid", "latents_fine")
    }
    source_bin = pack_hi_nerv_archive(
        decoder,
        sd["latents_coarse"].clone(),
        sd["latents_mid"].clone(),
        sd["latents_fine"].clone(),
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
            "use_hierarchical_feature_grid": cfg.use_hierarchical_feature_grid,
            "use_convnext_blocks": cfg.use_convnext_blocks,
            "local_grid_levels": cfg.local_grid_levels,
            "local_grid_channels": cfg.local_grid_channels,
            "convnext_mlp_ratio": cfg.convnext_mlp_ratio,
            "convnext_kernel_size": cfg.convnext_kernel_size,
        },
        decoder_codec="int8_mixed",
    )
    source_sections = split_archive_sections(source_bin)
    source = _source_zip(tmp_path / "source_hi_nerv.zip", source_bin)

    report = sweep_compact_decoder_codecs(
        source_archive_zip=source,
        output_dir=tmp_path / "sweep_hi_nerv",
        decoder_codecs=("int8_mixed", "int4_mixed"),
        repo_root=Path.cwd(),
        run_receiver_proof=False,
    )

    assert report["family"] == "hi_nerv"
    assert len(report["variant_rows"]) == 2
    assert report["byte_price_plan"]["candidate_id"] == (
        "hi_nerv_compact_decoder_codec_sweep"
    )
    assert report["byte_price_plan"]["score_claim"] is False
    for row in report["variant_rows"]:
        assert row["promotion_eligible"] is False
        assert row["charged_bits_changed"] is True
        assert "receiver_proof_not_run" in row["blockers"]
        parsed = parse_hi_nerv_archive(Path(row["bin_path"]).read_bytes())
        sections = split_archive_sections(Path(row["bin_path"]).read_bytes())
        assert sections.latents_coarse_blob == source_sections.latents_coarse_blob
        assert sections.latents_mid_blob == source_sections.latents_mid_blob
        assert sections.latents_fine_blob == source_sections.latents_fine_blob
        assert parsed.meta["_decoder_state_codec"]["codec"] == row["parsed_decoder_codec"][
            "codec"
        ]


def test_replay_adjudication_preserves_pact_vq_rate_primitive(
    tmp_path: Path,
) -> None:
    source_profile = _profile_with_response(
        tmp_path,
        "source",
        _mlx_response(
            score=90.66354296056916,
            d_seg=0.5048259229958058,
            d_pose=161.237585550944,
            rate=0.02655045989679346,
            bytes_=39_874,
        ),
    )
    best_profile = _profile_with_response(
        tmp_path,
        "best",
        _mlx_response(
            score=90.66201548013069,
            d_seg=0.5048259229958058,
            d_pose=161.237585550944,
            rate=0.0250229794583312,
            bytes_=37_580,
        ),
    )
    sweep = {
        "schema": "compact_decoder_codec_sweep.v1",
        "family": "pact_nerv_vq",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "full_video_mlx_scorer_replay_not_attached",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "best_variant": {
            "decoder_codec": "int2_mixed",
            "archive_bytes": 37_641,
            "archive_sha256": "sha-int2",
            "receiver_proof_passed": True,
            "blockers": [
                "full_video_mlx_scorer_replay_not_attached",
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
        },
    }

    adjudicated = adjudicate_compact_decoder_codec_sweep_with_replay(
        codec_sweep_report=sweep,
        source_replay_profile=source_profile,
        best_codec_replay_profile=best_profile,
    )

    replay = adjudicated["replay_adjudication"]
    assert replay["schema"] == "compact_decoder_codec_replay_adjudication.v1"
    assert replay["gate_verdict"] == "PRESERVE_RATE_PRIMITIVE_EXACT_BLOCKED_BY_DISTORTION"
    assert replay["preserve_rate_primitive"] is True
    assert replay["exact_axis_blocked"] is True
    assert replay["exact_spend_candidate"] is False
    assert replay["ready_for_exact_eval_dispatch"] is False
    assert replay["score_claim"] is False
    assert "full_video_mlx_scorer_replay_not_attached" not in adjudicated["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in adjudicated["blockers"]
    assert adjudicated["best_variant"]["full_video_mlx_replay_attached"] is True
    assert adjudicated["best_variant"]["preserve_rate_primitive"] is True
    assert adjudicated["ready_for_exact_eval_dispatch"] is False


def _profile_with_response(tmp_path: Path, name: str, response: dict) -> dict:
    response_path = tmp_path / f"{name}_response.json"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return {
        "schema": "hprc_mlx_component_neutralization_profile.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "variant_rows": [
            {
                "variant_id": "baseline",
                "archive_zip_bytes": response["archive_size_bytes"],
                "mlx_response": response_path.as_posix(),
            }
        ],
    }


def _mlx_response(
    *,
    score: float,
    d_seg: float,
    d_pose: float,
    rate: float,
    bytes_: int,
) -> dict:
    return {
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "avg_segnet_dist": d_seg,
        "avg_posenet_dist": d_pose,
        "score_rate_contribution": rate,
        "archive_size_bytes": bytes_,
        "evidence_tag": "[macOS-MLX research-signal]",
        "score_axis": "[macOS-MLX research-signal]",
        "max_pairs": 600,
        "n_samples": 600,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
