# SPDX-License-Identifier: MIT
"""Regression tests for Check 39: undeployed archive-artifact producers.

Catches the recurring "code-shipped-never-deployed" failure mode (Lane EC sat
unused 2 weeks; TIER 3 lanes still follow this pattern).

Reference: project_lane_ec_engineered_corrections_20260428.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_undeployed_archive_artifact_producers,
    _producer_has_main_entry,
    _scan_repo_for_archive_artifact_producers,
)
from tac.source_index import source_index_context


def _setup_fake_repo(root: Path) -> None:
    """Build a minimal repo skeleton: experiments/, src/tac/, scripts/."""
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)


def test_strict_passes_on_real_codebase() -> None:
    """The live repo must pass this check — no undeployed producers."""
    violations = check_undeployed_archive_artifact_producers(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"check_undeployed_archive_artifact_producers found {len(violations)} "
        f"violation(s) — fix the deployment gap or add to "
        f"_DEPLOY_SCANNER_EXEMPT_PRODUCERS with WHY comment. Violations:\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_detects_undeployed_producer(tmp_path: Path) -> None:
    """A producer with __main__ writing a known artifact, no remote_lane_*.sh
    referencing it → must be flagged."""
    _setup_fake_repo(tmp_path)
    producer = tmp_path / "experiments" / "make_corrections.py"
    producer.write_text(textwrap.dedent('''\
        """Producer that writes corrections.bin."""
        import argparse

        def main():
            with open("corrections.bin", "wb") as f:
                f.write(b"data")

        if __name__ == "__main__":
            main()
    '''))
    violations = check_undeployed_archive_artifact_producers(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "make_corrections.py" in violations[0]
    assert "corrections.bin" in violations[0]
    assert "code-shipped-never-deployed" in violations[0]


def test_passes_when_remote_lane_references_basename(tmp_path: Path) -> None:
    """Producer is deployed iff a scripts/remote_lane_*.sh mentions it."""
    _setup_fake_repo(tmp_path)
    producer = tmp_path / "experiments" / "make_corrections.py"
    producer.write_text(textwrap.dedent('''\
        import argparse
        with open("corrections.bin", "wb") as f:
            f.write(b"data")
        if __name__ == "__main__":
            pass
    '''))
    deploy = tmp_path / "scripts" / "remote_lane_test.sh"
    deploy.write_text("#!/bin/bash\npython experiments/make_corrections.py\n")
    violations = check_undeployed_archive_artifact_producers(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_passes_when_remote_lane_references_artifact_name(tmp_path: Path) -> None:
    """Inline-producer pattern: deploy script writes artifact filename
    directly via `python -c`. Producer is still considered deployed."""
    _setup_fake_repo(tmp_path)
    producer = tmp_path / "experiments" / "make_corrections.py"
    producer.write_text(textwrap.dedent('''\
        with open("corrections.bin", "wb") as f:
            f.write(b"data")
        if __name__ == "__main__":
            pass
    '''))
    deploy = tmp_path / "scripts" / "remote_lane_test.sh"
    deploy.write_text(
        "#!/bin/bash\n"
        "python -c \"...\" > corrections.bin\n"
    )
    violations = check_undeployed_archive_artifact_producers(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_library_files_without_main_are_skipped(tmp_path: Path) -> None:
    """Files writing artifacts but lacking __main__ are libraries — exempt."""
    _setup_fake_repo(tmp_path)
    lib = tmp_path / "src" / "tac" / "corrections_lib.py"
    lib.write_text(textwrap.dedent('''\
        def write_corrections(path):
            with open("corrections.bin", "wb") as f:
                f.write(b"data")
    '''))
    violations = check_undeployed_archive_artifact_producers(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_strict_raises_metabugviolation(tmp_path: Path) -> None:
    """strict=True with violations → MetaBugViolation."""
    _setup_fake_repo(tmp_path)
    producer = tmp_path / "experiments" / "make_corrections.py"
    producer.write_text(textwrap.dedent('''\
        with open("corrections.bin", "wb") as f:
            f.write(b"")
        if __name__ == "__main__":
            pass
    '''))
    with pytest.raises(MetaBugViolation, match="UNDEPLOYED ARCHIVE-ARTIFACT"):
        check_undeployed_archive_artifact_producers(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_artifact_producer_basename_match(tmp_path: Path) -> None:
    """Helper: _producer_has_main_entry detects the canonical pattern."""
    _setup_fake_repo(tmp_path)
    p = tmp_path / "with_main.py"
    p.write_text("if __name__ == \"__main__\":\n    pass\n")
    assert _producer_has_main_entry(p) is True
    q = tmp_path / "no_main.py"
    q.write_text("def f():\n    pass\n")
    assert _producer_has_main_entry(q) is False


def test_deploy_registry_counts_as_deployed(tmp_path: Path) -> None:
    """A producer referenced from src/tac/deploy/**/*.py is deployed
    (covers train_joint_pair.py invoked through deploy_vastai.experiments)."""
    _setup_fake_repo(tmp_path)
    (tmp_path / "src" / "tac" / "deploy" / "vastai").mkdir(parents=True, exist_ok=True)
    producer = tmp_path / "experiments" / "make_corrections.py"
    producer.write_text(textwrap.dedent('''\
        with open("corrections.bin", "wb") as f:
            f.write(b"")
        if __name__ == "__main__":
            pass
    '''))
    registry = tmp_path / "src" / "tac" / "deploy" / "vastai" / "experiments.py"
    registry.write_text(
        '''EXPERIMENTS = {"corrections": "experiments/make_corrections.py"}\n'''
    )
    violations = check_undeployed_archive_artifact_producers(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_archive_artifact_producer_scan_is_single_pass_under_source_index(
    tmp_path: Path,
) -> None:
    """The Check 39 producer discovery should not rescan once per artifact."""

    _setup_fake_repo(tmp_path)
    first = tmp_path / "experiments" / "make_renderer.py"
    first.write_text(textwrap.dedent('''\
        def main():
            with open("renderer.bin", "wb") as f:
                f.write(b"data")

        if __name__ == "__main__":
            main()
    '''))
    second = tmp_path / "experiments" / "make_masks.py"
    second.write_text(textwrap.dedent('''\
        def main():
            with open("masks.mkv", "wb") as f:
                f.write(b"data")

        if __name__ == "__main__":
            main()
    '''))
    irrelevant = tmp_path / "src" / "tac" / "library.py"
    irrelevant.write_text("VALUE = 'renderer.bin but no write marker'\n")

    with source_index_context(tmp_path) as index:
        producers = _scan_repo_for_archive_artifact_producers(
            tmp_path,
            source_index=index,
        )
        stats = index.stats()

    assert producers["renderer.bin"] == (first,)
    assert producers["masks.mkv"] == (second,)
    assert producers["optimized_poses.bin"] == ()
    assert stats["file_list_misses"] == 1
    assert stats["text_misses"] == 3
