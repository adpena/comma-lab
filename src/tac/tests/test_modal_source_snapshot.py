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


def test_the_snapshot_is_injected_by_env_not_by_changing_the_cwd() -> None:
    """The regression guard for the ps2 rc=5 refusal.

    Firing with ``cwd`` inside the snapshot made the worker's local half spawn a RELATIVE
    ``.venv/bin/python`` that a snapshot does not contain — and, had it resolved, would
    have written the dispatch claim into the snapshot's own ``.omx/state``. The snapshot
    must therefore reach Modal through the mount root, never through the cwd.

    The REAL app modules are loaded, both ways, because the property under test is what
    the dispatcher actually does — not what its source says.
    """

    import importlib.util
    import subprocess
    import sys

    from tac.modal_source_snapshot import SOURCE_ROOT_ENV

    assert dispatch_env(Path("/snap"), {}, entrypoint=Path("/snap/x/w.py"))[SOURCE_ROOT_ENV] == "/snap"

    repo = Path(__file__).resolve().parents[3]
    if importlib.util.find_spec("modal") is None:  # pragma: no cover - env without modal
        import pytest

        pytest.skip("modal is not installed; the app modules cannot be loaded")
    probe = (
        "import importlib.util,sys\n"
        "sys.path.insert(0,'src'); sys.path.insert(0,'experiments')\n"
        "spec=importlib.util.spec_from_file_location('m', sys.argv[1])\n"
        "m=importlib.util.module_from_spec(spec); sys.modules['m']=m; spec.loader.exec_module(m)\n"
        "print(m._mount_path('src'))\n"
    )
    for name in ("modal_auth_eval", "modal_auth_eval_cpu"):
        module = str(repo / "experiments" / f"{name}.py")
        plain = subprocess.run(
            [sys.executable, "-c", probe, module], cwd=repo, capture_output=True, text=True,
            env={k: v for k, v in __import__("os").environ.items() if k != SOURCE_ROOT_ENV},
        )
        assert plain.returncode == 0, plain.stderr[-800:]
        assert plain.stdout.strip() == "src", f"{name} is not a no-op with the env unset"

        redirected = subprocess.run(
            [sys.executable, "-c", probe, module], cwd=repo, capture_output=True, text=True,
            env={**__import__("os").environ, SOURCE_ROOT_ENV: "/snap"},
        )
        assert redirected.returncode == 0, redirected.stderr[-800:]
        assert redirected.stdout.strip() == "/snap/src", f"{name} ignores {SOURCE_ROOT_ENV}"


def test_mount_paths_survive_the_helper_wrapper(tmp_path: Path) -> None:
    module = tmp_path / "app.py"
    module.write_text(
        'import modal\n'
        'image = (modal.Image.debian_slim()\n'
        '  .add_local_dir(_mount_path("src"), remote_path="/w/src")\n'
        '  .add_local_file(_mount_path("uv.lock"), remote_path="/w/uv.lock"))\n'
    )
    mounts = extract_mount_paths(module)
    assert mounts.dirs == ("src",)
    assert mounts.files == ("uv.lock",)


def test_dispatch_path_guard_catches_the_ps2_failure() -> None:
    """Every relative path must resolve from the dispatch cwd, checked before the meter."""

    from tac.modal_source_snapshot import local_relative_spawn_paths, verify_dispatch_paths

    repo = Path(__file__).resolve().parents[3]
    spawned = local_relative_spawn_paths()
    assert ".venv/bin/python" in spawned
    assert "tools/claim_lane_dispatch.py" in spawned
    assert verify_dispatch_paths(repo, ["experiments/modal_auth_eval.py::main"]) == []
    problems = verify_dispatch_paths(repo / ".omx/tmp/a_snapshot_that_is_not_a_repo_root", [])
    assert any(".venv/bin/python" in p for p in problems), problems
