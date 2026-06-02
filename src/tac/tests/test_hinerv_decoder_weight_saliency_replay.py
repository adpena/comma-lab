# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.analysis.hinerv_decoder_weight_saliency_replay import (
    HinervDecoderWeightSaliencyReplayError,
    build_hinerv_decoder_weight_saliency_replay,
    write_hinerv_decoder_weight_saliency_replay,
)
from tac.framework_agnostic.helpers import write_npz_bridge_artifact
from tac.repo_io import sha256_file
from tac.substrates.hi_nerv.architecture import HinervConfig, HinervSubstrate
from tac.substrates.hi_nerv.archive import pack_archive


def test_hinerv_decoder_weight_saliency_replay_emits_real_gradient_rows(
    tmp_path: Path,
) -> None:
    ladder = _fixture_ladder(tmp_path)

    report = build_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=ladder,
        row_ids=("fixture_tiny",),
        video_path=tmp_path / "fixture_not_used.mkv",
        upstream_dir=tmp_path / "fixture_upstream_not_used",
        device="cpu",
        max_pairs=1,
        scorer_loader=_fixture_scorer_loader,
        pair_loader=_fixture_pair_loader,
        scorer_source="fixture_not_score_authority",
    )

    assert report["schema"] == "hinerv_decoder_weight_saliency_replay.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    assert report["rows"][0]["row_id"] == "fixture_tiny"
    assert report["rows"][0]["sampled_pairs"] == 1
    assert "full_video_coverage_missing" in report["blockers"]
    assert report["saliency_rows"]
    assert all("decoder_weight_saliency" in row for row in report["saliency_rows"])
    assert any(value > 0.0 for value in report["saliency_by_name"].values())
    assert "latents_coarse" not in report["saliency_by_name"]
    assert report["eval_roundtrip_applied_to_predictions"] is True
    assert report["rows"][0]["eval_roundtrip_applied_to_predictions"] is True


def test_hinerv_decoder_weight_saliency_replay_applies_eval_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import differentiable_eval_roundtrip

    calls: list[tuple[int, ...]] = []

    def fake_roundtrip(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        calls.append(tuple(int(dim) for dim in x.shape))
        return x

    monkeypatch.setattr(
        differentiable_eval_roundtrip,
        "apply_eval_roundtrip_during_training",
        fake_roundtrip,
    )
    ladder = _fixture_ladder(tmp_path)

    build_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=ladder,
        row_ids=("fixture_tiny",),
        video_path=tmp_path / "fixture_not_used.mkv",
        upstream_dir=tmp_path / "fixture_upstream_not_used",
        device="cpu",
        max_pairs=1,
        scorer_loader=_fixture_scorer_loader,
        pair_loader=_fixture_pair_loader,
        scorer_source="fixture_not_score_authority",
    )

    assert calls == [(1, 3, 4, 4), (1, 3, 4, 4)]


def test_hinerv_decoder_weight_saliency_replay_blocks_unfit_allocator_basin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import differentiable_eval_roundtrip

    monkeypatch.setattr(
        differentiable_eval_roundtrip,
        "apply_eval_roundtrip_during_training",
        lambda x, *args, **kwargs: x,
    )
    ladder = _fixture_ladder(tmp_path)

    report = build_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=ladder,
        row_ids=("fixture_tiny",),
        video_path=tmp_path / "fixture_not_used.mkv",
        upstream_dir=tmp_path / "fixture_upstream_not_used",
        device="cpu",
        max_pairs=1,
        scorer_loader=_fixture_scorer_loader,
        pair_loader=_fixture_pair_loader,
        scorer_source="fixture_not_score_authority",
        max_mean_score_loss_proxy_for_allocator=1.0e-12,
    )

    row = report["rows"][0]
    assert row["loss_summary"]["allocator_linearization_basin_passed"] is False
    assert "score_loss_proxy_outside_allocator_linearization_basin" in row["blockers"]
    assert "score_loss_proxy_outside_allocator_linearization_basin" in report["blockers"]


def test_hinerv_decoder_weight_saliency_replay_preserves_official_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tac import differentiable_eval_roundtrip

    monkeypatch.setattr(
        differentiable_eval_roundtrip,
        "apply_eval_roundtrip_during_training",
        lambda x, *args, **kwargs: x,
    )
    cfg = HinervConfig(
        latent_dim_coarse=2,
        latent_dim_mid=2,
        latent_dim_fine=2,
        embed_dim=4,
        initial_grid_h=1,
        initial_grid_w=1,
        decoder_channels=(4, 3),
        sin_frequency=3.0,
        num_upsample_blocks=2,
        mid_injection_block_index=0,
        fine_injection_block_index=1,
        num_pairs=2,
        output_height=4,
        output_width=4,
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=2,
        local_grid_channels=2,
        convnext_mlp_ratio=2,
        convnext_kernel_size=3,
    )
    ladder = _fixture_ladder(tmp_path, cfg=cfg)

    report = build_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=ladder,
        row_ids=("fixture_tiny",),
        video_path=tmp_path / "fixture_not_used.mkv",
        upstream_dir=tmp_path / "fixture_upstream_not_used",
        device="cpu",
        max_pairs=1,
        scorer_loader=_fixture_scorer_loader,
        pair_loader=_fixture_pair_loader,
        scorer_source="fixture_not_score_authority",
    )

    assert report["rows"][0]["saliency_by_name"]
    assert any("feature_grids.0.grids.0" in key for key in report["saliency_by_name"])
    assert any("convnext_blocks.0.dwconv.weight" in key for key in report["saliency_by_name"])


def test_hinerv_decoder_weight_saliency_replay_fails_closed_on_archive_sha(
    tmp_path: Path,
) -> None:
    ladder = _fixture_ladder(tmp_path)
    ladder["archive_rows"][0]["archive_sha256"] = "0" * 64

    with pytest.raises(
        HinervDecoderWeightSaliencyReplayError,
        match="archive sha mismatch",
    ):
        build_hinerv_decoder_weight_saliency_replay(
            archive_ladder_report=ladder,
            row_ids=("fixture_tiny",),
            video_path=tmp_path / "fixture_not_used.mkv",
            upstream_dir=tmp_path / "fixture_upstream_not_used",
            device="cpu",
            max_pairs=1,
            scorer_loader=_fixture_scorer_loader,
            pair_loader=_fixture_pair_loader,
            scorer_source="fixture_not_score_authority",
        )


def test_write_hinerv_decoder_weight_saliency_replay_outputs_json_and_md(
    tmp_path: Path,
) -> None:
    ladder = _fixture_ladder(tmp_path)
    output_json = tmp_path / "saliency.json"
    output_md = tmp_path / "saliency.md"

    report = write_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=ladder,
        row_ids=("fixture_tiny",),
        video_path=tmp_path / "fixture_not_used.mkv",
        upstream_dir=tmp_path / "fixture_upstream_not_used",
        device="cpu",
        max_pairs=1,
        scorer_loader=_fixture_scorer_loader,
        pair_loader=_fixture_pair_loader,
        scorer_source="fixture_not_score_authority",
        output_json=output_json,
        output_md=output_md,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["report_path"] == str(output_json)
    assert payload["saliency_by_name"] == report["saliency_by_name"]
    assert "HiNeRV decoder-weight saliency replay" in output_md.read_text(
        encoding="utf-8"
    )


class _FixturePoseScorer(torch.nn.Module):
    fixture_not_score_authority = True

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        first = flat[:, :1]
        pose3 = torch.cat([mean, std, first], dim=1)
        return {"pose": pose3.repeat(1, 4)}


class _FixtureSegScorer(torch.nn.Module):
    fixture_not_score_authority = True

    def preprocess_input(self, pair_btchw: torch.Tensor) -> torch.Tensor:
        return pair_btchw[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        return torch.cat(
            [
                mean,
                -mean,
                x[:, 0:1] / 255.0,
                x[:, 1:2] / 255.0,
                x[:, 2:3] / 255.0,
            ],
            dim=1,
        )


def _fixture_scorer_loader(
    _upstream_dir: Path,
    _device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    return _FixturePoseScorer(), _FixtureSegScorer()


def _fixture_pair_loader(
    _video_path: Path,
    max_pairs: int,
    _start_pair: int,
    _pair_stride: int,
    target_hw: tuple[int, int],
) -> list[torch.Tensor]:
    h, w = target_hw
    base = torch.linspace(0, 255, steps=2 * 3 * h * w, dtype=torch.float32)
    pair = base.reshape(1, 2, 3, h, w)
    return [pair.clone() for _ in range(max_pairs)]


def _fixture_ladder(tmp_path: Path, *, cfg: HinervConfig | None = None) -> dict:
    if cfg is None:
        cfg = HinervConfig(
            latent_dim_coarse=2,
            latent_dim_mid=2,
            latent_dim_fine=2,
            embed_dim=4,
            initial_grid_h=1,
            initial_grid_w=1,
            decoder_channels=(4, 3),
            sin_frequency=3.0,
            num_upsample_blocks=2,
            mid_injection_block_index=0,
            fine_injection_block_index=1,
            num_pairs=2,
            output_height=4,
            output_width=4,
        )
    torch.manual_seed(7)
    model = HinervSubstrate(cfg)
    state = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.state_dict().items()
    }
    state_npz = tmp_path / "hi_nerv_mlx_exported_state.npz"
    manifest_path = tmp_path / "hi_nerv_mlx_exported_state_npz_manifest.json"
    write_npz_bridge_artifact(
        state,
        state_npz,
        source_backend="mlx",
        bridge_kind="hi_nerv_mlx_export_state_dict_to_npz",
        manifest_path=manifest_path,
        require_finite=True,
    )
    sd = model.state_dict()
    decoder_sd = {
        k: v.detach().clone()
        for k, v in sd.items()
        if k not in {"latents_coarse", "latents_mid", "latents_fine"}
    }
    meta = {
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
    }
    blob = pack_archive(
        decoder_sd,
        sd["latents_coarse"].detach().clone(),
        sd["latents_mid"].detach().clone(),
        sd["latents_fine"].detach().clone(),
        meta,
    )
    archive_zip = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", blob)
    return {
        "schema": "hinerv_archive_size_ladder.v1",
        "family": "hi_nerv",
        "report_path": str(tmp_path / "ladder.json"),
        "num_pairs": cfg.num_pairs,
        "archive_rows": [
            {
                "row_id": "fixture_tiny",
                "archive_path": str(archive_zip),
                "archive_sha256": sha256_file(archive_zip),
                "archive_bytes": archive_zip.stat().st_size,
                "state_npz_manifest_path": str(manifest_path),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
