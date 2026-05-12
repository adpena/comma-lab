"""Tests for ``tac.deploy.modal.mount_manifest``.

Coverage targets per the T1-A landing memo:
- structural minimum is ALWAYS mounted (skip_structural=False default)
- trainer introspection discovers ``required_input_file=True`` flags
- missing ``required_input_file`` default → MountManifestError (fail-closed)
- ``TIER_1_EXTRA_MOUNT_PATHS`` is honored (both forms: tuple and list)
- ``MODAL_EXTRA_MOUNT_PATHS`` is honored as a sibling alias
- optional_files / optional_dirs are mounted iff present
- multi-tier (TIER_1 + TIER_2) flag manifests are unioned
- non-dict TIER_<N>_OPERATOR_REQUIRED_FLAGS attributes are ignored
- trainer module path resolution: file path AND dotted module name
- missing trainer module → MountManifestError
- missing structural minimum directory → MountManifestError
- absolute paths in defaults resolve correctly
- remote_repo override threads through
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from tac.deploy.modal.mount_manifest import (
    DEFAULT_REMOTE_REPO,
    MountManifestError,
    MountUploadRaceError,
    build_training_image,
    collect_extra_mount_paths,
    collect_tier_required_input_files,
    verify_mount_set_mtime_stability,
)


@pytest.fixture(autouse=True)
def _disable_mtime_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mtime-stability check (Catalog #165) sleeps 2s per call. Disable
    it for the existing mount-manifest test module; the new
    ``test_check_165_modal_mount_mtime_stability.py`` module tests it
    explicitly."""

    monkeypatch.setenv("TAC_MODAL_MTIME_STABILITY_DISABLED", "1")


class FakeImage:
    """Records add_local_dir / add_local_file calls so tests can assert mounts."""

    def __init__(self) -> None:
        self.dirs: list[tuple[str, str, list[str] | None]] = []
        self.files: list[tuple[str, str]] = []

    def add_local_dir(
        self, local: str, *, remote_path: str, ignore: list[str] | None = None
    ) -> FakeImage:
        self.dirs.append((local, remote_path, ignore))
        return self

    def add_local_file(self, local: str, *, remote_path: str) -> FakeImage:
        self.files.append((local, remote_path))
        return self


@pytest.fixture(autouse=True)
def _disable_mtime_stability_wait_for_unit_tests(monkeypatch: pytest.MonkeyPatch):
    """Unit tests use fake Modal images; avoid production upload-race sleeps."""
    monkeypatch.setenv("TAC_MODAL_MTIME_STABILITY_DISABLED", "1")


def _make_fake_repo(tmp_path: Path) -> Path:
    """Build a minimal repo skeleton with all structural minimums present."""

    root = tmp_path / "fakerepo"
    for d in ("src", "scripts", "upstream", "submissions", "experiments", "tools"):
        (root / d).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='fake'\n")
    return root


def _make_trainer_module(
    tmp_path: Path,
    *,
    name: str = "fake_trainer",
    body: str = "",
) -> Path:
    """Write a minimal trainer module file with the supplied body."""

    path = tmp_path / f"{name}.py"
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Structural minimum
# ---------------------------------------------------------------------------


def test_structural_minimum_always_mounted(tmp_path: Path) -> None:
    """All 6 structural dirs + pyproject.toml are mounted by default."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    result = build_training_image(image, repo_root=root)

    mounted_dirs = {rel for rel, _, _ in result.dirs}
    assert mounted_dirs == {"src", "scripts", "upstream", "submissions", "experiments", "tools"}
    mounted_files = {rel for rel, _ in result.files}
    assert mounted_files == {"pyproject.toml"}


def test_structural_minimum_uses_default_remote_repo(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    result = build_training_image(image, repo_root=root)
    for rel, remote, _ in result.dirs:
        assert remote == f"{DEFAULT_REMOTE_REPO}/{rel}"


def test_remote_repo_override(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    result = build_training_image(image, repo_root=root, remote_repo="/alt/remote")
    for rel, remote, _ in result.dirs:
        assert remote == f"/alt/remote/{rel}"


def test_experiments_dir_has_results_ignore(tmp_path: Path) -> None:
    """``experiments/`` mounts with ``ignore=['results/**']``."""

    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    result = build_training_image(image, repo_root=root)
    by_rel = {rel: ignore for rel, _, ignore in result.dirs}
    assert by_rel["experiments"] == ["results/**"]
    assert by_rel["src"] is None
    assert by_rel["tools"] is None


def test_skip_structural_omits_minimum(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    result = build_training_image(image, repo_root=root, skip_structural=True)
    assert result.dirs == []
    assert result.files == []


def test_missing_structural_minimum_dir_raises(tmp_path: Path) -> None:
    """If ``tools/`` is missing on disk the builder fails closed."""

    root = _make_fake_repo(tmp_path)
    (root / "tools").rmdir()
    image = FakeImage()
    with pytest.raises(MountManifestError, match="structural minimum directory"):
        build_training_image(image, repo_root=root)


def test_missing_structural_minimum_file_raises(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "pyproject.toml").unlink()
    image = FakeImage()
    with pytest.raises(MountManifestError, match="structural minimum file"):
        build_training_image(image, repo_root=root)


# ---------------------------------------------------------------------------
# Operator extras
# ---------------------------------------------------------------------------


def test_extra_dirs_appended(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "data").mkdir()
    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        extra_dirs=[("data", None)],
    )
    assert ("data", f"{DEFAULT_REMOTE_REPO}/data", None) in result.dirs


def test_extra_files_appended(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "manifest.json").write_text("{}")
    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        extra_files=["manifest.json"],
    )
    assert ("manifest.json", f"{DEFAULT_REMOTE_REPO}/manifest.json") in result.files


def test_extra_dirs_with_ignore(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "results").mkdir()
    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        extra_dirs=[("results", ("raw/**", "tmp/**"))],
    )
    by_rel = {rel: ignore for rel, _, ignore in result.dirs}
    assert by_rel["results"] == ["raw/**", "tmp/**"]


def test_optional_files_present_mounted(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "license.txt").write_text("MIT")
    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        optional_files=["license.txt", "missing.txt"],
    )
    files = {rel for rel, _ in result.files}
    assert "license.txt" in files
    assert "missing.txt" not in files


def test_optional_dirs_present_mounted(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "data").mkdir()
    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        optional_dirs=["data", "nodata"],
    )
    dirs = {rel for rel, _, _ in result.dirs}
    assert "data" in dirs
    assert "nodata" not in dirs


# ---------------------------------------------------------------------------
# Trainer introspection
# ---------------------------------------------------------------------------


def test_trainer_introspection_required_input_files_mounted(tmp_path: Path) -> None:
    """A ``required_input_file=True`` flag with an existing default is mounted."""

    root = _make_fake_repo(tmp_path)
    (root / ".omx" / "research").mkdir(parents=True)
    profile = root / ".omx/research/parity_profile.json"
    profile.write_text("{}")

    trainer = _make_trainer_module(
        root / "experiments",
        name="fake_trainer",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--profile': {\n"
            "        'env': 'PROFILE',\n"
            "        'rationale': 'test',\n"
            "        'default': '.omx/research/parity_profile.json',\n"
            "        'required_input_file': True,\n"
            "    },\n"
            "}\n"
        ),
    )

    image = FakeImage()
    result = build_training_image(
        image,
        repo_root=root,
        trainer_module_path=trainer,
    )
    files = {rel for rel, _ in result.files}
    assert ".omx/research/parity_profile.json" in files


def test_trainer_introspection_missing_required_input_file_raises(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--profile': {\n"
            "        'env': 'PROFILE',\n"
            "        'rationale': 'test',\n"
            "        'default': '.omx/research/never_generated.json',\n"
            "        'required_input_file': True,\n"
            "    },\n"
            "}\n"
        ),
    )
    image = FakeImage()
    with pytest.raises(MountManifestError, match="required_input_file defaults missing"):
        build_training_image(image, repo_root=root, trainer_module_path=trainer)


def test_trainer_required_input_file_without_default_is_skipped(tmp_path: Path) -> None:
    """Flags marked required but lacking a ``default`` are silently skipped."""

    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--profile': {\n"
            "        'env': 'PROFILE',\n"
            "        'rationale': 'test',\n"
            "        'required_input_file': True,\n"
            "    },\n"
            "}\n"
        ),
    )
    image = FakeImage()
    # Should not raise (no default means caller must thread an absolute override).
    build_training_image(image, repo_root=root, trainer_module_path=trainer)


def test_trainer_required_false_flags_ignored(tmp_path: Path) -> None:
    """Flags without ``required_input_file=True`` are NOT mounted."""

    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--batch-size': {\n"
            "        'env': 'BS',\n"
            "        'rationale': 'test',\n"
            "        'default': '32',\n"
            "    },\n"
            "}\n"
        ),
    )
    image = FakeImage()
    result = build_training_image(
        image, repo_root=root, trainer_module_path=trainer
    )
    # Only the structural minimum file.
    assert {rel for rel, _ in result.files} == {"pyproject.toml"}


def test_trainer_multi_tier_unioned(tmp_path: Path) -> None:
    """TIER_1 + TIER_2 required-input flags are both honored."""

    root = _make_fake_repo(tmp_path)
    (root / ".omx" / "research").mkdir(parents=True)
    (root / ".omx/research/p1.json").write_text("{}")
    (root / ".omx/research/p2.json").write_text("{}")

    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--p1': {'env': 'P1', 'rationale': 't', 'default': '.omx/research/p1.json', 'required_input_file': True},\n"
            "}\n"
            "TIER_2_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--p2': {'env': 'P2', 'rationale': 't', 'default': '.omx/research/p2.json', 'required_input_file': True},\n"
            "}\n"
        ),
    )
    image = FakeImage()
    result = build_training_image(
        image, repo_root=root, trainer_module_path=trainer
    )
    files = {rel for rel, _ in result.files}
    assert ".omx/research/p1.json" in files
    assert ".omx/research/p2.json" in files


def test_trainer_non_dict_manifest_ignored(tmp_path: Path) -> None:
    """A TIER_1_OPERATOR_REQUIRED_FLAGS that is not a dict is ignored without error."""

    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body="TIER_1_OPERATOR_REQUIRED_FLAGS = ['not', 'a', 'dict']\n",
    )
    image = FakeImage()
    build_training_image(image, repo_root=root, trainer_module_path=trainer)


def test_trainer_non_dict_meta_entry_ignored(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--profile': 'not-a-dict',\n"
            "}\n"
        ),
    )
    image = FakeImage()
    build_training_image(image, repo_root=root, trainer_module_path=trainer)


def test_trainer_extra_mount_paths_honored(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "fixtures").mkdir()
    (root / "fixtures" / "data.bin").write_bytes(b"x" * 8)
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_EXTRA_MOUNT_PATHS = ('fixtures', 'fixtures/data.bin')\n"
        ),
    )
    image = FakeImage()
    result = build_training_image(
        image, repo_root=root, trainer_module_path=trainer
    )
    dirs = {rel for rel, _, _ in result.dirs}
    files = {rel for rel, _ in result.files}
    assert "fixtures" in dirs
    assert "fixtures/data.bin" in files


def test_modal_extra_mount_paths_alias_honored(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    (root / "fixture.txt").write_text("x")
    trainer = _make_trainer_module(
        root / "experiments",
        body="MODAL_EXTRA_MOUNT_PATHS = ['fixture.txt']\n",
    )
    image = FakeImage()
    result = build_training_image(
        image, repo_root=root, trainer_module_path=trainer
    )
    files = {rel for rel, _ in result.files}
    assert "fixture.txt" in files


def test_trainer_extra_mount_paths_missing_raises(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        body="TIER_1_EXTRA_MOUNT_PATHS = ('does/not/exist.bin',)\n",
    )
    image = FakeImage()
    with pytest.raises(MountManifestError, match="TIER_1_EXTRA_MOUNT_PATHS"):
        build_training_image(image, repo_root=root, trainer_module_path=trainer)


def test_trainer_module_file_not_found_raises(tmp_path: Path) -> None:
    root = _make_fake_repo(tmp_path)
    image = FakeImage()
    with pytest.raises(MountManifestError, match="trainer module file not found"):
        build_training_image(
            image,
            repo_root=root,
            trainer_module_path=root / "experiments/no_such_trainer.py",
        )


def test_collect_helpers_directly() -> None:
    """``collect_*`` helpers operate on a constructed mock module."""

    mod = MagicMock()
    # Reset auto-discovered attrs.
    del mod.MODAL_EXTRA_MOUNT_PATHS
    mod.TIER_1_OPERATOR_REQUIRED_FLAGS = {
        "--p": {"required_input_file": True, "default": "foo.json"},
        "--q": {"required_input_file": False, "default": "bar.json"},
    }
    mod.TIER_2_OPERATOR_REQUIRED_FLAGS = {
        "--r": {"required_input_file": True, "default": "baz.json"},
    }
    mod.TIER_1_EXTRA_MOUNT_PATHS = ("extra/one", "extra/two")
    # MagicMock auto-creates attrs; replace with real lists by setting dir() shape.
    # We force dir() to return only the attrs we care about via a real module.

    # Use a tiny synthetic namespace instead.
    class NS:
        pass

    ns = NS()
    ns.TIER_1_OPERATOR_REQUIRED_FLAGS = mod.TIER_1_OPERATOR_REQUIRED_FLAGS
    ns.TIER_2_OPERATOR_REQUIRED_FLAGS = mod.TIER_2_OPERATOR_REQUIRED_FLAGS
    ns.TIER_1_EXTRA_MOUNT_PATHS = mod.TIER_1_EXTRA_MOUNT_PATHS

    required = collect_tier_required_input_files(ns)
    flags = {flag for flag, _ in required}
    assert flags == {"--p", "--r"}

    extras = collect_extra_mount_paths(ns)
    assert [str(p) for p in extras] == ["extra/one", "extra/two"]


def test_extras_dedup() -> None:
    """``TIER_1_EXTRA_MOUNT_PATHS`` + ``MODAL_EXTRA_MOUNT_PATHS`` are unioned with dedup."""

    class NS:
        TIER_1_EXTRA_MOUNT_PATHS = ("a", "b")
        MODAL_EXTRA_MOUNT_PATHS = ("b", "c")

    extras = collect_extra_mount_paths(NS())
    assert [str(p) for p in extras] == ["a", "b", "c"]


def test_trainer_introspection_does_not_pollute_sys_modules(tmp_path: Path) -> None:
    """File-path import goes through ``spec_from_file_location``; no sys.modules entry."""

    root = _make_fake_repo(tmp_path)
    trainer = _make_trainer_module(
        root / "experiments",
        name="zzz_introspect_only",
        body="TIER_1_OPERATOR_REQUIRED_FLAGS = {}\n",
    )
    image = FakeImage()
    build_training_image(image, repo_root=root, trainer_module_path=trainer)
    # importlib.util.module_from_spec doesn't insert into sys.modules by default.
    assert "zzz_introspect_only" not in sys.modules
    assert "_trainer_introspect_zzz_introspect_only" not in sys.modules


def test_collect_required_input_files_empty_default_skipped() -> None:
    class NS:
        TIER_1_OPERATOR_REQUIRED_FLAGS: ClassVar[dict[str, dict[str, object]]] = {
            "--p": {"required_input_file": True, "default": "   "},
            "--q": {"required_input_file": True, "default": 42},
            "--r": {"required_input_file": True, "default": None},
        }

    assert collect_tier_required_input_files(NS()) == []


def test_skip_structural_with_trainer_introspection(tmp_path: Path) -> None:
    """``skip_structural=True`` still honors trainer introspection."""

    root = _make_fake_repo(tmp_path)
    (root / ".omx" / "research").mkdir(parents=True)
    (root / ".omx/research/p.json").write_text("{}")
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            "    '--p': {'env': 'P', 'rationale': 't', 'default': '.omx/research/p.json', 'required_input_file': True},\n"
            "}\n"
        ),
    )
    image = FakeImage()
    result = build_training_image(
        image, repo_root=root, trainer_module_path=trainer, skip_structural=True
    )
    files = {rel for rel, _ in result.files}
    assert files == {".omx/research/p.json"}
    assert result.dirs == []


def test_default_repo_root_resolution() -> None:
    """When ``repo_root`` is omitted, the module default is used."""

    from tac.deploy.modal.mount_manifest import REPO_ROOT

    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "tac").is_dir()


def test_absolute_default_path_resolves(tmp_path: Path) -> None:
    """Trainer ``default`` may be an absolute path outside the repo root."""

    root = _make_fake_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    profile = outside / "abs_profile.json"
    profile.write_text("{}")
    trainer = _make_trainer_module(
        root / "experiments",
        body=(
            "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
            f"    '--p': {{'env': 'P', 'rationale': 't', 'default': '{profile}', 'required_input_file': True}},\n"
            "}\n"
        ),
    )
    image = FakeImage()
    # Should not raise.
    build_training_image(image, repo_root=root, trainer_module_path=trainer)


# ---------------------------------------------------------------------------
# Catalog #165: Modal mount upload-race protection
# ---------------------------------------------------------------------------


def test_verify_mount_set_mtime_stability_passes_for_stable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAC_MODAL_MTIME_STABILITY_DISABLED", raising=False)
    f = tmp_path / "stable.py"
    f.write_text("x = 1\n")

    verify_mount_set_mtime_stability(
        [f],
        window_seconds=0.0,
        max_retries=1,
        sleep_fn=lambda _seconds: None,
    )


def test_verify_mount_set_mtime_stability_fails_on_active_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAC_MODAL_MTIME_STABILITY_DISABLED", raising=False)
    f = tmp_path / "racing.py"
    f.write_text("x = 1\n")

    def mutate(_seconds: float) -> None:
        f.write_text(f.read_text() + "x += 1\n")

    with pytest.raises(MountUploadRaceError, match="fingerprint is unstable"):
        verify_mount_set_mtime_stability(
            [f],
            window_seconds=0.0,
            max_retries=2,
            sleep_fn=mutate,
        )
