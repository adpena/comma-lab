# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import torch

from tools.materialize_taskspace_fresh_teacher import _batch_preparer, _parser


class _BatchSensitiveSegNet:
    def __init__(self) -> None:
        self.forward_batch_sizes: list[int] = []

    def preprocess_input(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, 1]

    def __call__(self, scorer_input: torch.Tensor) -> torch.Tensor:
        self.forward_batch_sizes.append(int(scorer_input.shape[0]))
        rows = int(scorer_input.shape[0])
        logits = torch.zeros((rows, 5, *scorer_input.shape[-2:]), dtype=torch.float32)
        for row in range(rows):
            logits[row, row % 5] = 1.0
        return logits


def test_cli_defaults_to_frozen_upstream_evaluate_batch_geometry() -> None:
    assert _parser().parse_args(["--mode", "preflight"]).batch_size == 16


def test_partial_resume_forwards_full_original_batch_then_selects_missing_rows() -> None:
    segnet = _BatchSensitiveSegNet()
    source = np.zeros((4, 2, 2, 3, 3), dtype=np.uint8)
    prepared = _batch_preparer(segnet)(source)

    recovered = prepared.infer_missing((1, 3))

    assert segnet.forward_batch_sizes == [4]
    assert sorted(recovered) == [1, 3]
    assert np.all(recovered[1] == 1)
    assert np.all(recovered[3] == 3)
