"""Tests for Catalog #153 ``check_modal_dispatcher_uses_canonical_mount_builder``.

Refuses ``experiments/modal_*.py`` files that hand-curate Modal mounts
instead of routing through
``tac.deploy.modal.mount_manifest.build_training_image``.

Coverage:
- positive (canonical builder used → clean)
- positive (manual mount with same-line waiver → clean)
- negative (manual mount without waiver → violation)
- canonical builder used + extra manual mount without waiver → violation
- exempt files are excluded (mount_manifest.py + its test)
- comment-lines containing add_local_* are ignored
- non-experiments/modal_* paths are out of scope
- empty experiments/ directory → no scan, no violations
- live repo state is 0 violations (regression guard)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_dispatcher_uses_canonical_mount_builder,
)


def _make_repo(tmp_path: Path) -> Path:
    """Create a fake repo with an experiments/ directory."""

    root = tmp_path / "fakerepo"
    (root / "experiments").mkdir(parents=True)
    return root


def test_canonical_builder_only_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_a.py").write_text(
        "from tac.deploy.modal.mount_manifest import build_training_image\n"
        "image = build_training_image(base, trainer_module_path=None)\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_manual_mount_without_waiver_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_b.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_dir('src', remote_path='/x')\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "modal_b.py" in violations[0]
    assert "manual mount call" in violations[0]


def test_manual_mount_with_same_line_waiver_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_c.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_dir('src', remote_path='/x')  # MODAL_MANUAL_MOUNT_OK:narrow dispatcher\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_mixed_canonical_and_manual_needs_per_line_waiver(tmp_path: Path) -> None:
    """File using ``build_training_image`` PLUS an unwaived manual mount → violation."""

    root = _make_repo(tmp_path)
    (root / "experiments/modal_mix.py").write_text(
        "from tac.deploy.modal.mount_manifest import build_training_image\n"
        "image = build_training_image(base)\n"
        "image = image.add_local_dir('extra', remote_path='/y')\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "alongside canonical" in violations[0]


def test_mixed_canonical_and_waived_manual_clean(tmp_path: Path) -> None:
    """File using ``build_training_image`` + waived extra mount → clean."""

    root = _make_repo(tmp_path)
    (root / "experiments/modal_mix2.py").write_text(
        "from tac.deploy.modal.mount_manifest import build_training_image\n"
        "image = build_training_image(base)\n"
        "image = image.add_local_dir('extra', remote_path='/y')  # MODAL_MANUAL_MOUNT_OK:runtime-discovered payload\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_comment_line_with_token_ignored(tmp_path: Path) -> None:
    """A comment line that mentions ``.add_local_dir`` is not a violation."""

    root = _make_repo(tmp_path)
    (root / "experiments/modal_d.py").write_text(
        "from tac.deploy.modal.mount_manifest import build_training_image\n"
        "# We previously used image.add_local_dir(...) here.\n"
        "image = build_training_image(base)\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_non_modal_file_out_of_scope(tmp_path: Path) -> None:
    """A file under experiments/ that does not match modal_*.py is ignored."""

    root = _make_repo(tmp_path)
    (root / "experiments/something.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_dir('src', remote_path='/x')\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_no_experiments_dir_returns_empty(tmp_path: Path) -> None:
    """If ``experiments/`` does not exist (e.g. test repo), the check returns []."""

    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_strict_raises_on_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_violator.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_dir('src', remote_path='/x')\n"
    )
    with pytest.raises(PreflightError, match="manual Modal mount call"):
        check_modal_dispatcher_uses_canonical_mount_builder(
            repo_root=root, strict=True
        )


def test_strict_returns_empty_when_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_clean.py").write_text(
        "from tac.deploy.modal.mount_manifest import build_training_image\n"
        "image = build_training_image(base)\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=True
    )
    assert violations == []


def test_multiple_files_aggregated(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/modal_a.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_dir('a', remote_path='/a')\n"
    )
    (root / "experiments/modal_b.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_file('b.txt', remote_path='/b')\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert len(violations) == 2
    files = {v.split(":")[0] for v in violations}
    assert "experiments/modal_a.py" in files
    assert "experiments/modal_b.py" in files


def test_add_local_file_violation(tmp_path: Path) -> None:
    """``add_local_file`` (singular) is the same bug class as ``add_local_dir``."""

    root = _make_repo(tmp_path)
    (root / "experiments/modal_file.py").write_text(
        "import modal\n"
        "image = modal.Image.debian_slim().add_local_file('foo.txt', remote_path='/foo')\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_waiver_on_multiline_call(tmp_path: Path) -> None:
    """Waiver applies to the line containing ``.add_local_dir(``."""

    root = _make_repo(tmp_path)
    (root / "experiments/modal_multiline.py").write_text(
        "import modal\n"
        "image = (\n"
        "    modal.Image.debian_slim()\n"
        "    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:multi-line is fine\n"
        "        'src',\n"
        "        remote_path='/x',\n"
        "    )\n"
        ")\n"
    )
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=root, strict=False
    )
    assert violations == []


def test_live_repo_strict_passes() -> None:
    """Regression guard: the real repo passes the strict gate at landing."""

    repo_root = Path(__file__).resolve().parents[3]
    # Only assert if the experiments/ dir exists on disk (CI clones the
    # repo before running tests).
    if not (repo_root / "experiments").is_dir():
        pytest.skip("experiments/ not present on disk")
    violations = check_modal_dispatcher_uses_canonical_mount_builder(
        repo_root=repo_root, strict=False
    )
    assert violations == [], (
        f"Catalog #153 live violations (count={len(violations)}); first: "
        f"{violations[0] if violations else '<none>'}"
    )
