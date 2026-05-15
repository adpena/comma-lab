# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "experiments")
import train_substrate_pretrained_driving_prior as trainer


def test_trainer_rejects_comma10k_as_dp1_video_source() -> None:
    args = trainer.build_argparser().parse_args(
        ["--dataset-name", "comma10k", "--device", "cpu"]
    )

    with pytest.raises(SystemExit, match="semantic-segmentation image dataset"):
        trainer._validate_dataset_source_args(args, codebook_path=None)


def test_trainer_rejects_comma10k19_alias_as_dp1_video_source() -> None:
    args = trainer.build_argparser().parse_args(
        ["--dataset-name", "comma10k19", "--device", "cpu"]
    )

    with pytest.raises(SystemExit, match="semantic-segmentation image dataset"):
        trainer._validate_dataset_source_args(args, codebook_path=None)


def test_trainer_builds_dynamic_stream_chunking_strategy() -> None:
    args = trainer.build_argparser().parse_args(
        [
            "--dataset-name",
            "comma2k19",
            "--use-streamer",
            "--device",
            "cpu",
            "--stream-chunking-mode",
            "saliency",
            "--stream-saliency-topk",
            "3",
            "--stream-frame-range-size",
            "64",
        ]
    )

    strategy = trainer._build_dynamic_chunking_strategy(args)

    assert strategy.mode == "saliency"
    assert strategy.saliency_topk == 3
    assert strategy.frame_range_size == 64
