# SPDX-License-Identifier: MIT
"""Tests for the scorer-faithful PR95 Stage-8 public-archive lane."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    HNeRVSyntheticTrainingBundleMLX,
    pytorch_state_dict_from_mlx,
    write_pr95_public_archive_zip,
)
from tools.run_pr95_stage8_from_public_archive import (  # noqa: E402
    LANE_ID,
    PR95_STAGE8_COMPARISON_SCHEMA,
    PR95_STAGE8_LANE_SCHEMA,
    PR95_STAGE8_SEED_SCHEMA,
    build_compact_byte_grammar_reference,
    prepare_stage8_seed_from_archive,
    run_pr95_stage8_from_public_archive,
)


def _write_synthetic_pr95_archive(tmp_path: Path) -> Path:
    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=8,
        seed=41,
    )
    archive_zip = tmp_path / "source_archive.zip"
    write_pr95_public_archive_zip(
        pytorch_state_dict_from_mlx(bundle.decoder),
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=archive_zip,
    )
    return archive_zip


def test_prepare_stage8_seed_from_archive_writes_stage8_expected_files(
    tmp_path: Path,
) -> None:
    import torch

    source_archive_zip = _write_synthetic_pr95_archive(tmp_path)

    seed = prepare_stage8_seed_from_archive(
        source_archive_zip=source_archive_zip,
        output_dir=tmp_path / "run",
        overwrite=True,
    )

    assert seed.manifest["schema"] == PR95_STAGE8_SEED_SCHEMA
    assert seed.manifest["lane_id"] == LANE_ID
    assert seed.decoder_pt.is_file()
    assert seed.latents_pt.is_file()
    assert seed.bundle_pt.is_file()
    assert seed.manifest_path.is_file()
    assert seed.manifest["score_claim"] is False
    assert seed.manifest["promotion_eligible"] is False
    assert seed.manifest["ready_for_exact_eval_dispatch"] is False

    decoder_state = torch.load(seed.decoder_pt, weights_only=True, map_location="cpu")
    latents = torch.load(seed.latents_pt, weights_only=True, map_location="cpu")
    bundle = torch.load(seed.bundle_pt, weights_only=True, map_location="cpu")

    assert isinstance(decoder_state, dict)
    assert tuple(latents.shape) == (1, 28)
    assert "latents" in bundle
    assert tuple(bundle["latents"].shape) == (1, 28)


def test_plan_only_lane_report_blocks_proxy_promotion_and_writes_comparison_spine(
    tmp_path: Path,
) -> None:
    source_archive_zip = _write_synthetic_pr95_archive(tmp_path)

    report = run_pr95_stage8_from_public_archive(
        source_archive_zip=source_archive_zip,
        public_submission_root=tmp_path / "public_submission_root",
        challenge_root=tmp_path / "challenge",
        source_video_path=tmp_path / "challenge/videos/0.mkv",
        output_dir=tmp_path / "run",
        epochs=5000,
        eval_every=25,
        batch_size=8,
        muon_weight_decay=5e-4,
        device="cpu",
        execute=False,
        overwrite=True,
    )

    assert report["schema"] == PR95_STAGE8_LANE_SCHEMA
    assert report["mode"] == "plan_only"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "stage8_training_not_executed_plan_only" in report["exact_gate"][
        "blockers"
    ]
    assert "contest_cpu_cuda_exact_eval_missing" in report["exact_gate"]["blockers"]
    assert Path(report["report_path"]).is_file()

    comparison = json.loads(
        Path(report["compact_base_renderer_byte_grammar_path"]).read_text()
    )
    assert comparison["schema"] == PR95_STAGE8_COMPARISON_SCHEMA
    assert comparison["families"][0]["family"] == (
        "pr95_hnerv_stage8_from_public_archive"
    )
    assert comparison["families"][0]["executable_now"] is True
    assert all(row["proxy_promotion_allowed"] is False for row in comparison["families"])
    assert any(row["family"] == "rnerv" for row in comparison["families"])
    assert any(row["family"] == "rt_vq_nerv" for row in comparison["families"])


def test_compact_byte_grammar_reference_requires_runtime_custody() -> None:
    comparison = build_compact_byte_grammar_reference(
        pr95_report={
            "report_path": "/tmp/report.json",
            "candidate_archive_zip_bytes": 178309,
            "exact_gate": {"blockers": ["contest_cpu_cuda_exact_eval_missing"]},
        }
    )

    assert comparison["schema"] == PR95_STAGE8_COMPARISON_SCHEMA
    assert comparison["score_claim"] is False
    assert comparison["promotion_eligible"] is False
    for row in comparison["families"]:
        assert "trained_decoder_weights_or_program" in row["required_sections"]
        assert "receiver_proof" in row["required_sections"]
        assert "exact_gate_or_blocker" in row["required_sections"]
        assert row["hard_byte_ceilings"] == [100000, 178417, 216000, 285000]
