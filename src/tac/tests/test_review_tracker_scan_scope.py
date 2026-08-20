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


def test_recovered_ssd_evidence_is_out_of_review_scope() -> None:
    """Verbatim SSD-recovered snapshots are archived evidence, not maintained source.

    ddm_sd1 landed ~600 of them so the bytes stop living on one disk. Scanning them would
    register unreviewed entities that block every commit, and the only escapes would be to
    fake-review code nobody read or to override the gate on `.py` — both forbidden. Excluding
    them keeps the gate counting code we maintain.
    """
    assert not review_tracker._is_reviewable_python_path(
        "experiments/ssd_recovered/APDataStore/ddm_sa1/builders/compose.py"
    )
    assert not review_tracker._is_reviewable_python_path(
        "experiments/ssd_recovered/VertigoDataTier/evidence/x/runner.py"
    )
    # The exclusion is scoped to that one directory: a sibling under experiments/ still scans,
    # so this cannot be widened by accident into "experiments is unreviewed".
    assert review_tracker._is_reviewable_python_path("experiments/ssd_recovery_tool.py")
    assert review_tracker._is_reviewable_python_path("experiments/pipeline.py")


def test_review_tracker_required_source_roots_are_self_protected() -> None:
    assert review_tracker.REQUIRED_SOURCE_SCAN_PREFIXES == (
        "src/tac/",
        "src/comma_lab/",
    )
    assert review_tracker.review_tracker_scan_scope_blockers() == []


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


def test_extract_entities_disambiguates_rebound_module_names(tmp_path: Path) -> None:
    module_path = tmp_path / "sample.py"
    module_path.write_text(
        "def load():\n"
        "    return 1\n\n"
        "def load():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    original_root = review_tracker.REPO_ROOT
    try:
        review_tracker.REPO_ROOT = tmp_path
        entities = review_tracker.extract_entities(module_path, compute_complexity=False)
    finally:
        review_tracker.REPO_ROOT = original_root

    assert [entity.name for entity in entities] == ["load", "load"]
    assert [entity.qualified_name for entity in entities] == [
        "sample::load@L1",
        "sample::load@L4",
    ]
