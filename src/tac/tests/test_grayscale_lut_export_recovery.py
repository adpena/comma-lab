# SPDX-License-Identifier: MIT
"""Regression tests for grayscale_lut timeout/export recovery hardening."""

from __future__ import annotations

import subprocess
import zipfile
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_grayscale_lut.py"
DRIVER_PATH = REPO_ROOT / "scripts" / "remote_lane_substrate_grayscale_lut.sh"
RECIPES_DIR = REPO_ROOT / ".omx" / "operator_authorize_recipes"


def test_cli_accepts_export_only_checkpoint_and_soft_deadline() -> None:
    import experiments.train_substrate_grayscale_lut as trainer

    args = trainer._build_parser().parse_args(
        [
            "--video-path",
            "upstream/videos/0.mkv",
            "--output-dir",
            "/tmp/grayscale_lut_recovery",
            "--epochs",
            "2000",
            "--device",
            "cuda",
            "--export-only-checkpoint",
            "/tmp/harvested/best.pt",
            "--soft-train-deadline-seconds",
            "12600",
        ]
    )

    assert args.export_only_checkpoint == Path("/tmp/harvested/best.pt")
    assert args.soft_train_deadline_seconds == 12600


def test_export_archive_from_checkpoint_builds_archive_without_training(
    tmp_path: Path,
) -> None:
    import torch

    import experiments.train_substrate_grayscale_lut as trainer
    from tac.substrates.grayscale_lut.architecture import (
        GrayscaleLutConfig,
        GrayscaleLutSubstrate,
    )

    cfg = GrayscaleLutConfig(
        grayscale_downsample=2,
        decoder_hidden=4,
        decoder_blocks=1,
        embedding_dim=2,
        num_pairs=2,
        output_height=8,
        output_width=8,
        lut_bits=5,
    )
    model = GrayscaleLutSubstrate(cfg)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "config": asdict(cfg),
        },
        checkpoint_path,
    )

    args = Namespace(output_dir=tmp_path / "out", skip_archive_build=False)
    result = trainer._export_archive_from_checkpoint(args, checkpoint_path)

    assert result["archive_bytes"] > 0
    assert len(result["archive_sha256"]) == 64
    assert (args.output_dir / "0.bin").is_file()
    assert (args.output_dir / "submission" / "inflate.sh").is_file()
    assert (args.output_dir / "archive.zip").is_file()
    with zipfile.ZipFile(args.output_dir / "archive.zip") as zf:
        assert sorted(zf.namelist()) == ["0.bin", "inflate.py", "inflate.sh"]


def test_remote_driver_threads_export_recovery_env_and_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(DRIVER_PATH)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

    body = DRIVER_PATH.read_text(encoding="utf-8")
    assert "GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT" in body
    assert "--export-only-checkpoint" in body
    assert "GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS" in body
    assert "--soft-train-deadline-seconds" in body
    assert '${TRAINER_EXTRA_ARGS[@]+"${TRAINER_EXTRA_ARGS[@]}"}' in body
    assert "contest_auth_eval_cpu.json" in body


@pytest.mark.parametrize(
    "recipe_name",
    [
        "substrate_grayscale_lut_modal_a100_dispatch.yaml",
        "substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch.yaml",
        "substrate_grayscale_lut_lut_bits_5_local_mlx_dispatch.yaml",
    ],
)
def test_grayscale_lut_recipes_propagate_export_recovery_env(
    recipe_name: str,
) -> None:
    recipe = yaml.safe_load((RECIPES_DIR / recipe_name).read_text(encoding="utf-8"))
    env = recipe["env_overrides"]

    assert env["GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT"] == (
        "${GRAYSCALE_LUT_EXPORT_ONLY_CHECKPOINT:-}"
    )
    assert int(env["GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS"]) > 0
    assert int(env["GRAYSCALE_LUT_SOFT_TRAIN_DEADLINE_SECONDS"]) < (
        float(recipe["timeout_hours"]) * 3600
    )


def test_trainer_source_mentions_export_only_recovery_flag() -> None:
    body = TRAINER_PATH.read_text(encoding="utf-8")
    assert "--export-only-checkpoint" in body
    assert "training_stopped_by_soft_deadline" in body
