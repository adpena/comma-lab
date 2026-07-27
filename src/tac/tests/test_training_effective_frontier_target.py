# SPDX-License-Identifier: MIT
"""Regression tests for the trainer's competitive frontier target.

The training controller must route against the best qualifying local or
official-upstream score.  A local-CPU-only read silently weakens that target
whenever the public leaderboard is ahead.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tac.canonical_frontier_pointer import (
    FrontierPointerCorruptError,
    effective_frontier_score,
    load_canonical_frontier_pointer_strict,
)

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments" / "train_witness_realized_through_R_mlx.py"


def _load_trainer():
    name = "_training_effective_frontier_target"
    spec = importlib.util.spec_from_file_location(name, TRAINER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_pointer(repo_root: Path, payload: dict) -> None:
    path = repo_root / ".omx" / "state" / "canonical_frontier_pointer.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_module_frontier_equals_live_effective_pointer() -> None:
    trainer = _load_trainer()
    pointer = load_canonical_frontier_pointer_strict(repo_root=REPO)
    expected = effective_frontier_score(pointer)
    assert expected is not None
    assert trainer.FRONTIER == expected


def test_loader_recomposes_minimum_and_ignores_stale_cached_winner(tmp_path: Path) -> None:
    trainer = _load_trainer()
    payload = json.loads(
        (REPO / ".omx" / "state" / "canonical_frontier_pointer.json").read_text(
            encoding="utf-8"
        )
    )
    payload["our_local_frontier_contest_cpu"]["score"] = 0.31
    payload["our_local_frontier_contest_cuda"]["score"] = 0.29
    payload["upstream_leaderboard_snapshot"]["best_entry"]["score"] = 0.17
    payload["upstream_leaderboard_snapshot"]["entries"][0]["score"] = 0.17
    payload["effective_frontier"]["score"] = 0.99
    _write_pointer(tmp_path, payload)

    trainer.REPO = tmp_path
    assert trainer._load_frontier() == pytest.approx(0.17)


def test_loader_fails_closed_when_pointer_is_missing(tmp_path: Path) -> None:
    trainer = _load_trainer()
    trainer.REPO = tmp_path
    with pytest.raises(FrontierPointerCorruptError, match="pointer missing"):
        trainer._load_frontier()
