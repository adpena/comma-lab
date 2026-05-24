# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tools import review_tracker


def _rel(path: Path) -> str:
    return path.relative_to(review_tracker.REPO_ROOT).as_posix()


def test_review_tracker_scan_scope_excludes_custody_mirrors() -> None:
    assert review_tracker._is_reviewable_python_path("src/tac/preflight.py")
    assert review_tracker._is_reviewable_python_path(
        "src/comma_lab/scheduler/ssh_experiment_queue_executor.py"
    )
    assert review_tracker._is_reviewable_python_path("experiments/train_renderer.py")
    assert review_tracker._is_reviewable_python_path("tools/review_tracker.py")
    assert review_tracker._is_reviewable_python_path("submissions/robust_current/inflate.py")

    assert not review_tracker._is_reviewable_python_path(
        "experiments/results/public_pr103_intake_20260504_codex/source/inflate.py"
    )
    assert not review_tracker._is_reviewable_python_path("reports/raw/generated.py")
    assert not review_tracker._is_reviewable_python_path("upstream/evaluate.py")
    assert not review_tracker._is_reviewable_python_path("docs/example.py")
    assert not review_tracker._is_reviewable_python_path("tools/__init__.py")


def test_git_ls_files_output_is_deduped_and_deterministic() -> None:
    paths = review_tracker._reviewable_python_paths_from_git_output(
        "\n".join(
            [
                "experiments/results/mirror/source/train.py",
                "tools/review_tracker.py",
                "src/comma_lab/scheduler/staircase_dag.py",
                "src/tac/preflight.py",
                "src/tac/preflight.py",
                "reports/raw/generated.py",
                "submissions/robust_current/inflate.py",
            ]
        )
    )

    assert [_rel(path) for path in paths] == [
        "src/comma_lab/scheduler/staircase_dag.py",
        "src/tac/preflight.py",
        "submissions/robust_current/inflate.py",
        "tools/review_tracker.py",
    ]


def test_extract_entities_can_skip_complexity_walk(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def f():\n"
        "    if a:\n"
        "        for x in y:\n"
        "            return x\n",
        encoding="utf-8",
    )

    original_root = review_tracker.REPO_ROOT
    try:
        review_tracker.REPO_ROOT = tmp_path
        fast = review_tracker.extract_entities(module_path, compute_complexity=False)
        full = review_tracker.extract_entities(module_path, compute_complexity=True)
    finally:
        review_tracker.REPO_ROOT = original_root

    assert fast[0].complexity == 1
    assert full[0].complexity > fast[0].complexity
