# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "measure_rep_mine_solved_binary.py"
    spec = importlib.util.spec_from_file_location("measure_rep_mine_solved_binary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _module()


def test_entropy_and_conditional_entropy() -> None:
    assert M.entropy_bits(np.array([2, 2])) == 4.0
    counts = np.array([[2, 0], [1, 1]])
    assert M.conditional_entropy_bits(counts) == 2.0


def test_sha256_named_files_binds_names_and_content(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    expected = hashlib.sha256()
    for path in (first, second):
        expected.update(path.name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(hashlib.sha256(path.read_bytes()).digest())
    assert M.sha256_named_files([first, second]) == expected.hexdigest()


def test_boundary_mask_marks_both_sides() -> None:
    labels = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.uint8)
    expected = np.array([[False, True, True], [False, True, True]])
    assert np.array_equal(M.boundary_mask(labels), expected)


def test_run_length_summary_counts_each_axis() -> None:
    labels = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    row = M.run_length_summary(labels)
    assert row["horizontal_runs_by_class"].tolist() == [2, 2, 0, 0, 0]
    assert row["vertical_runs_by_class"].tolist() == [2, 2, 0, 0, 0]
    assert row["pixels_by_class"].tolist() == [3, 3, 0, 0, 0]
    assert row["horizontal_equal_adjacencies"] == 2
    assert row["vertical_equal_adjacencies"] == 2


def test_log2_choose_symmetry_and_endpoints() -> None:
    assert M.log2_choose(10, 0) == 0.0
    assert M.log2_choose(10, 10) == 0.0
    assert abs(M.log2_choose(10, 3) - M.log2_choose(10, 7)) < 1e-12


def test_fisher_exception_plan_detects_perfect_prefix() -> None:
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    row = M.fisher_exception_plan({a, b}, [a, b, c], 32)
    assert row["prefix"] == 2
    assert row["hits"] == 2
    assert row["deletions"] == 0
    assert row["additions"] == 0
    assert row["agreement_at_k"] == 1.0
