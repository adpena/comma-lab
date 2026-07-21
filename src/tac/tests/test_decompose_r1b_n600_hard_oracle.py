from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "decompose_r1b_n600_hard_oracle.py"
    spec = importlib.util.spec_from_file_location("decompose_r1b_n600_hard_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_summarize_label_batch_closes_integer_and_class_accounting() -> None:
    gt = np.array(
        [
            [[0, 1, 2], [3, 4, 0]],
            [[4, 3, 2], [1, 0, 4]],
        ],
        dtype=np.int64,
    )
    candidate = gt.copy()
    candidate[0, 0, 1] = 0
    candidate[0, 1, 1] = 3
    candidate[1, 0, 0] = 2

    row = tool.summarize_label_batch(gt, candidate)

    assert row["total_pixels"] == 12
    assert row["mismatch_pixels"] == 3
    assert row["pair_mismatch_pixels"] == [2, 1]
    assert sum(value["gt_pixels"] for value in row["per_class"].values()) == 12
    assert sum(value["mismatch_pixels"] for value in row["per_class"].values()) == 3
    assert row["per_class"]["Lane"]["mismatch_pixels"] == 1
    assert row["per_class"]["MyCar"]["mismatch_pixels"] == 2


def test_summarize_label_batch_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        tool.summarize_label_batch(
            np.zeros((1, 2, 2), dtype=np.int64),
            np.zeros((1, 3, 2), dtype=np.int64),
        )


def test_top_pair_rows_is_descending_and_stable() -> None:
    assert tool.top_pair_rows([0.2, 0.9, 0.9, 0.1], top_k=3) == [
        {"pair_index": 1, "value": 0.9},
        {"pair_index": 2, "value": 0.9},
        {"pair_index": 0, "value": 0.2},
    ]


def test_top_pair_rows_requires_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        tool.top_pair_rows([0.2], top_k=0)


def test_checkpoint_roundtrip_and_contract_refusal(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    first = {
        "schema": f"{tool.SCHEMA}.checkpoint.v1",
        "contract_sha256": "a" * 64,
        "batch_count": 1,
    }
    second = {**first, "batch_count": 2}
    tool._append_checkpoint(path, first)
    tool._append_checkpoint(path, second)
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    assert tool._load_checkpoint(path, contract_sha256="a" * 64) == second
    with pytest.raises(tool.DecompositionError, match="contract mismatch"):
        tool._load_checkpoint(path, contract_sha256="b" * 64)
