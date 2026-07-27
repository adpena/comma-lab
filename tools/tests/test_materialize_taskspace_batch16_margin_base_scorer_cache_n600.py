from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

TOOL_PATH = Path(__file__).resolve().parents[1] / "materialize_taskspace_batch16_margin_base_scorer_cache_n600.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("g78_margin_base_cli", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_winner_margin_is_winner_minus_maximum_nonwinner() -> None:
    tool = _load_tool()
    logits = torch.tensor(
        [
            [
                [[1.0, 4.0]],
                [[3.0, 4.0]],
                [[2.0, -1.0]],
                [[-5.0, 3.0]],
                [[0.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    cells, margins = tool._winner_fields(logits)
    np.testing.assert_array_equal(cells, np.asarray([[[1, 0]]], dtype=np.uint8))
    # Pixel 0: 3 - 2. Pixel 1 is a class-0/class-1 tie; torch.argmax selects
    # class 0 and the exact winner-minus-best-other margin is zero.
    np.testing.assert_array_equal(
        margins,
        np.asarray([[[1.0, 0.0]]], dtype=np.float32),
    )


def test_cli_materialize_asserts_governed_admission_before_run(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    tool = _load_tool()
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text("{}", encoding="utf-8")
    order: list[str] = []

    def admit(*_args, **_kwargs):
        order.append("admit")
        return True

    def run(_config):
        order.append("run")
        return aggregate, {"aggregate_receipt_sha256": "a" * 64}

    monkeypatch.setattr(tool, "assert_governed_admission", admit)
    monkeypatch.setattr(tool, "run_materialization", run)
    monkeypatch.setattr(
        tool,
        "file_identity",
        lambda path: {"path": str(path), "bytes": 2, "sha256": "b" * 64},
    )
    assert tool.main([str(tmp_path / "config.json"), "--materialize"]) == 0
    assert order == ["admit", "run"]
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["score_claim"] is False
    assert receipt["candidate_claim"] is False
    assert receipt["closed_only_after_aggregate"] == [
        "G72_FRESH_BATCH16_TARGET_MARGIN_CUSTODY_OWED",
        "G72_FRESH_V15_CAMERA_R_BATCH16_BASE_SCORER_STAGE_CACHE_OWED",
    ]


def test_cli_requires_explicit_action(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit):
        tool.main([str(tmp_path / "config.json")])


def test_runtime_closure_contains_transitive_v15_camera_dependencies() -> None:
    tool = _load_tool()
    rows = tool._runtime_sources()
    paths = [path for _role, path in rows]
    assert len(paths) == len(set(paths))
    relative = {str(path.relative_to(tool.REPO_ROOT)) for path in paths}
    assert "src/tac/optimization/direct_description_carrier_compose.py" in relative
    assert "src/tac/through_r/resolution_chain.py" in relative
    assert "src/tac/contest_eval_contract.py" in relative
