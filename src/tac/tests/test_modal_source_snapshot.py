# SPDX-License-Identifier: MIT
"""The Modal source snapshot: derived mounts, immunity, and fail-closed completeness.

The property under test is the one the ddm_fs2 refusal cost a re-fire for: after the
snapshot is taken, a working-tree edit must not be able to reach the bytes the image
build reads. Everything else here guards the ways that property can be silently lost —
a mount the AST missed, a short copy, an import that escapes the snapshot.
"""

from __future__ import annotations

from pathlib import Path

from tac.modal_source_snapshot import (
    build_snapshot,
    dispatch_env,
    extract_mount_paths,
    files_digest,
    iter_mount_files,
    prune_snapshots,
    verify_snapshot,
)

APP_MODULE = '''
import modal
image = (
    modal.Image.debian_slim()
    .add_local_dir("src", remote_path="/w/src", copy=True)
    .add_local_dir("upstream", remote_path="/w/upstream")
    .add_local_file("pyproject.toml", remote_path="/w/pyproject.toml")
    .add_local_python_source("worker", "tac")
)
'''


def _make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("x = 1\n")
    (root / "src" / "pkg" / "__pycache__").mkdir()
    (root / "src" / "pkg" / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    (root / "upstream").mkdir()
    (root / "upstream" / "evaluate.py").write_text("print('eval')\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "experiments").mkdir()
    (root / "experiments" / "worker.py").write_text(APP_MODULE)
    return root


def test_mount_paths_are_parsed_from_the_module_ast(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    mounts = extract_mount_paths(root / "experiments" / "worker.py")
    assert mounts.dirs == ("src", "upstream")
    assert mounts.files == ("pyproject.toml",)
    assert mounts.python_source_modules == ("tac", "worker")


def test_generated_files_are_excluded_the_way_modal_excludes_them(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    names = {p.as_posix() for p in iter_mount_files(root, "src")}
    assert "src/pkg/__init__.py" in names
    assert not any("__pycache__" in n for n in names)


def test_snapshot_is_immune_to_a_working_tree_edit_after_it_is_taken(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    snap = build_snapshot(
        source_root=root,
        entrypoint=root / "experiments" / "worker.py",
        snapshot_root=tmp_path / "snap",
    )
    assert snap.complete, snap.verify_failures
    before = snap.files_digest
    (root / "src" / "pkg" / "__init__.py").write_text("x = 2  # a formatter ran mid-build\n")
    after, _, _ = files_digest(snap.root, ["src", "upstream", "pyproject.toml", "experiments/worker.py"])
    assert after == before


def test_a_short_snapshot_refuses_instead_of_being_fired(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    snap = build_snapshot(
        source_root=root,
        entrypoint=root / "experiments" / "worker.py",
        snapshot_root=tmp_path / "snap2",
    )
    (snap.root / "upstream" / "evaluate.py").unlink()
    failures = verify_snapshot(root, snap.root, ["src", "upstream"])
    assert failures
    assert any("not in snapshot" in f for f in failures)


def test_a_mount_absent_from_the_source_is_reported_not_silently_dropped(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    snap = build_snapshot(
        source_root=root,
        entrypoint=root / "experiments" / "worker.py",
        snapshot_root=tmp_path / "snap3",
        extra_paths=("nothing/here",),
    )
    assert "nothing/here" in snap.missing_in_source
    assert not snap.complete


def test_dispatch_env_puts_the_snapshot_first_on_pythonpath(tmp_path: Path) -> None:
    env = dispatch_env(tmp_path / "snap", {"PYTHONPATH": "/elsewhere"},
                       entrypoint=tmp_path / "snap" / "experiments" / "worker.py")
    parts = env["PYTHONPATH"].split(":")
    assert parts[0].endswith("experiments")
    assert parts[1] == str(tmp_path / "snap" / "src")
    assert parts[-1] == "/elsewhere"


def test_pruning_removes_only_snapshots_older_than_the_retention(tmp_path: Path) -> None:
    root = tmp_path / "snaps"
    old = root / "old"
    new = root / "new"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    import os
    import time

    stale = time.time() - 10 * 86400
    os.utime(old, (stale, stale))
    removed = prune_snapshots(root, retain_days=3.0)
    assert removed == ["old"]
    assert new.is_dir()
    assert not old.exists()


def test_nested_mounts_are_not_double_counted_in_the_digest(tmp_path: Path) -> None:
    """A digest that shifts because ``src/tac`` also appears beside ``src`` is unusable.

    This is the wart the hv1 race self-test surfaced: the recorded digest and an honest
    re-digest disagreed only because one walk carried the redundant nested path.
    """

    from tac.modal_source_snapshot import dedupe_nested

    assert dedupe_nested(["src", "src/tac", "upstream", "src/tac/x.py"]) == ["src", "upstream"]
    root = _make_repo(tmp_path)
    with_nested, count_a, _ = files_digest(root, ["src", "src/pkg", "upstream"])
    without, count_b, _ = files_digest(root, ["src", "upstream"])
    assert with_nested == without
    assert count_a == count_b
