from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "build_a5_channel_qbits_schedule.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_a5_channel_qbits_schedule", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_dp_allocates_bits_to_high_loss_channels() -> None:
    tool = _load_tool()
    loss_table = np.full((3, 9), np.inf, dtype=np.float64)
    for dim, weights in enumerate(([100.0], [10.0], [1.0])):
        weight = weights[0]
        for bits in range(1, 9):
            loss_table[dim, bits] = weight * float(8 - bits) ** 2

    q_bits, loss = tool._solve_exact_dp(loss_table, target_qsum=18)

    assert q_bits.tolist() == [8, 7, 3]
    assert loss == 35.0


def test_channel_loss_table_zero_for_all8_and_monotone() -> None:
    tool = _load_tool()
    q = np.array(
        [
            [255, 128],
            [64, 32],
            [17, 8],
        ],
        dtype=np.uint8,
    )
    scales = np.array([1.0, 0.5], dtype=np.float64)

    table = tool._channel_loss_table(q, scales)

    assert table.shape == (2, 9)
    np.testing.assert_allclose(table[:, 8], 0.0)
    assert table[0, 4] >= table[0, 5] >= table[0, 6] >= table[0, 7] >= table[0, 8]


def test_build_schedule_records_no_score_claim(monkeypatch) -> None:
    tool = _load_tool()
    q_pair_first = np.array(
        [
            [255, 128, 17, 1],
            [64, 32, 16, 8],
        ],
        dtype=np.uint8,
    )
    scales = np.ones(4, dtype=np.float64)

    monkeypatch.setattr(
        tool,
        "_read_single_member",
        lambda _path: b"x" * (tool.PR101_DECODER_BLOB_LEN + tool.PR101_LATENT_BLOB_LEN),
    )
    monkeypatch.setattr(
        tool,
        "_extract_latents",
        lambda _member: (b"m" * 16, q_pair_first, scales),
    )

    payload = tool.build_schedule(
        source_archive=Path(__file__),
        target_qsum=24,
        repo_root=REPO_ROOT,
    )

    assert payload["schema"] == "pr101_a5_channel_qbits_schedule.v1"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert sum(payload["per_channel_q_bits"]) == 24
    assert payload["q_bits_sideinfo"]["encoding"] == "channel_raw3"
    assert payload["proxy_objective"]["optimal_for_target_qsum"] is True
