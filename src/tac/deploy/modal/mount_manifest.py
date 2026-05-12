"""Canonical Modal mount manifest builder.

The discovery-based pattern for Modal dispatch images. Replaces 11 hand-curated
mount lists across ``experiments/modal_*.py`` with one builder that:

1. ALWAYS mounts the structural minimum (``src/``, ``scripts/``, ``upstream/``,
   ``tools/``, ``submissions/``, ``experiments/`` w/ ``results/**`` ignored,
   ``pyproject.toml``).
2. Optionally imports the trainer module via ``importlib`` and reads its
   ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` manifests. Every flag declaring
   ``required_input_file=True`` resolves its default to a real path and
   adds an ``add_local_file`` mount.
3. Supports a module-level ``TIER_1_EXTRA_MOUNT_PATHS`` tuple on the trainer
   module for paths the trainer reads at runtime but that do not appear as
   CLI flags (fixture data, vendored prior probes, etc.).
4. Validates that every declared ``required_input_file`` resolves on disk
   BEFORE returning the image. Fail-closed at build-time, not at the GPU's
   first I/O.

Bug-class anchor: 2026-05-12 Modal A100 dispatch ``fc-01KREJST89QHFRWJXHAKXD850C``
crashed in 15s rc=1 because ``.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json``
(the trainer's default ``--pr95-parity-profile`` value) was not in the mount
list. The "fix" was to append it as the (N+1)th entry — adding to hand-curated
lists IS the bug class. This module replaces the list-of-strings with a
discovery-based contract; future flags marked ``required_input_file=True``
are automatically mounted.

Catalog #153 (``check_modal_dispatcher_uses_canonical_mount_builder``) refuses
any ``experiments/modal_*.py`` that calls ``Image.debian_slim(...).add_local_dir(...)``
or ``.add_local_file(...)`` directly without routing through
``tac.deploy.modal.mount_manifest.build_training_image``.

Cross-references:
- CLAUDE.md "Subagent coherence-by-default" non-negotiable (engineering primitives,
  not orchestration layers).
- Catalog #151 ``check_operator_wrapper_threads_trainer_tier_required_flags``
  (the wrapper-side sister of this builder — the trainer manifest IS the
  authoritative contract).
- Catalog #152 ``check_operator_wrapper_validates_required_input_files_pre_dispatch``
  (validates that ``required_input_file=True`` defaults exist; this module
  ensures they get mounted).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REMOTE_REPO = "/workspace/pact"
DEFAULT_RESULTS_IGNORE: tuple[str, ...] = ("results/**",)

# The structural minimum mount set. Every Modal training/eval dispatch needs
# these. Adding/removing entries here is a design decision that should be
# council-reviewed.
STRUCTURAL_MINIMUM_DIRS: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("src", None),
    ("scripts", None),
    ("upstream", None),
    ("submissions", None),
    ("experiments", DEFAULT_RESULTS_IGNORE),
    ("tools", None),
)

STRUCTURAL_MINIMUM_FILES: tuple[str, ...] = (
    "pyproject.toml",
)


class MountManifestError(RuntimeError):
    """Raised when the mount manifest builder cannot satisfy its contract.

    Notable subclasses:
    - missing-required-input-file
    - trainer-module-not-importable
    - declared-extra-mount-not-on-disk
    """


def _resolve_repo_path(rel_or_abs: str | Path, *, repo_root: Path) -> Path:
    """Resolve a path string relative to repo root unless already absolute."""

    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p)


def _import_trainer_module(trainer_module_path: str | Path) -> Any:
    """Import a trainer module by file path WITHOUT executing import-side effects.

    ``trainer_module_path`` may be a dotted module name (``experiments.train_foo``)
    OR a file path (``experiments/train_foo.py``). For file paths we use
    ``importlib.util.spec_from_file_location`` to avoid polluting ``sys.modules``.
    """

    path_str = str(trainer_module_path)
    if path_str.endswith(".py") or "/" in path_str or os.sep in path_str:
        file_path = _resolve_repo_path(path_str, repo_root=REPO_ROOT)
        if not file_path.is_file():
            raise MountManifestError(
                f"trainer module file not found: {file_path} "
                f"(resolved from {trainer_module_path!r})"
            )
        spec = importlib.util.spec_from_file_location(
            f"_trainer_introspect_{file_path.stem}", file_path
        )
        if spec is None or spec.loader is None:
            raise MountManifestError(
                f"could not build importlib spec for trainer module {file_path}"
            )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - surfaced at build time only
            raise MountManifestError(
                f"trainer module {file_path} raised at import time: {type(exc).__name__}: {exc}"
            ) from exc
        return module
    try:
        return importlib.import_module(path_str)
    except Exception as exc:
        raise MountManifestError(
            f"could not import trainer module {trainer_module_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def collect_tier_required_input_files(trainer_module: Any) -> list[tuple[str, Path]]:
    """Return ``[(flag_name, default_path), ...]`` for every required input file.

    Scans every ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` attribute on the module
    (unions them across N) and yields flags whose metadata declares
    ``required_input_file=True``. The ``default`` value must be a non-empty
    string path; flags missing it are skipped.
    """

    out: list[tuple[str, Path]] = []
    for attr_name in sorted(dir(trainer_module)):
        if not attr_name.startswith("TIER_"):
            continue
        if not attr_name.endswith("_OPERATOR_REQUIRED_FLAGS"):
            continue
        manifest = getattr(trainer_module, attr_name, None)
        if not isinstance(manifest, dict):
            continue
        for flag, meta in manifest.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("required_input_file") is not True:
                continue
            default = meta.get("default")
            if not isinstance(default, str) or not default.strip():
                continue
            out.append((str(flag), Path(default)))
    return out


def collect_extra_mount_paths(trainer_module: Any) -> list[Path]:
    """Return paths declared in ``TIER_1_EXTRA_MOUNT_PATHS`` (or sibling tuples).

    Trainers that need to mount fixture data / vendored probes / etc. that do
    NOT appear as CLI flags can declare a module-level tuple of strings or
    Paths under one of these conventional names:

    - ``TIER_1_EXTRA_MOUNT_PATHS``
    - ``MODAL_EXTRA_MOUNT_PATHS``

    Both forms are unioned. Entries may be relative (resolved against repo
    root) or absolute.
    """

    seen: set[str] = set()
    out: list[Path] = []
    for attr in ("TIER_1_EXTRA_MOUNT_PATHS", "MODAL_EXTRA_MOUNT_PATHS"):
        value = getattr(trainer_module, attr, None)
        if value is None:
            continue
        if not isinstance(value, (tuple, list)):
            continue
        for entry in value:
            if not isinstance(entry, (str, Path)):
                continue
            text = str(entry).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(Path(text))
    return out


def build_training_image(
    base_image: Any,
    *,
    trainer_module_path: str | Path | None = None,
    repo_root: Path | None = None,
    remote_repo: str = DEFAULT_REMOTE_REPO,
    extra_dirs: Iterable[tuple[str, tuple[str, ...] | None]] = (),
    extra_files: Iterable[str] = (),
    optional_dirs: Iterable[str] = (),
    optional_files: Iterable[str] = (),
    skip_structural: bool = False,
) -> Any:
    """Build a Modal Image with canonical structural mounts + trainer discovery.

    Parameters
    ----------
    base_image:
        A ``modal.Image`` (already configured with apt / pip / runtime). The
        builder appends ``add_local_dir`` / ``add_local_file`` calls and
        returns the new image. The base image is not mutated.
    trainer_module_path:
        Optional. If provided, the builder imports the module and discovers
        ``TIER_<N>_OPERATOR_REQUIRED_FLAGS`` + ``TIER_1_EXTRA_MOUNT_PATHS``.
        For every ``required_input_file=True`` flag, the default path MUST
        resolve on disk OR this function raises ``MountManifestError`` BEFORE
        returning the image (fail-closed at build-time).
    repo_root:
        Defaults to the package's own resolution. Override only for tests.
    remote_repo:
        Defaults to ``/workspace/pact``. Lane scripts assume this prefix.
    extra_dirs:
        Iterable of ``(local_path, ignore_globs_or_None)`` tuples appended
        after the structural minimum.
    extra_files:
        Iterable of relative file paths appended after the structural minimum
        files.
    optional_dirs:
        Iterable of relative dir paths that are mounted IFF they exist on
        disk. Used for paths that may be conditionally present.
    optional_files:
        Iterable of relative file paths that are mounted IFF they exist on
        disk.
    skip_structural:
        When True the structural minimum is NOT mounted. Use only for
        narrow specialised dispatchers (e.g. eval-only paths that mount a
        subset of ``upstream/``). The structural minimum is the default.

    Returns
    -------
    A new ``modal.Image`` ready to ``.run_function`` against.

    Raises
    ------
    MountManifestError:
        If trainer-declared ``required_input_file=True`` defaults do not exist
        on disk, or if trainer-declared ``EXTRA_MOUNT_PATHS`` entries are
        missing on disk.
    """

    root = (repo_root or REPO_ROOT).resolve()
    remote = Path(remote_repo)
    image = base_image

    # 1. Structural minimum.
    if not skip_structural:
        for rel, ignore in STRUCTURAL_MINIMUM_DIRS:
            local = root / rel
            if not local.is_dir():
                raise MountManifestError(
                    f"structural minimum directory missing on local repo: {local} "
                    f"(rel={rel!r}; this is a repo-state bug — every Modal dispatch "
                    "expects this directory present)"
                )
            remote_path = str(remote / rel)
            if ignore is not None:
                image = image.add_local_dir(rel, remote_path=remote_path, ignore=list(ignore))
            else:
                image = image.add_local_dir(rel, remote_path=remote_path)
        for rel in STRUCTURAL_MINIMUM_FILES:
            local = root / rel
            if not local.is_file():
                raise MountManifestError(
                    f"structural minimum file missing on local repo: {local} "
                    f"(rel={rel!r})"
                )
            image = image.add_local_file(rel, remote_path=str(remote / rel))

    # 2. Operator-declared extras.
    for rel, ignore in extra_dirs:
        remote_path = str(remote / rel)
        if ignore is not None:
            image = image.add_local_dir(rel, remote_path=remote_path, ignore=list(ignore))
        else:
            image = image.add_local_dir(rel, remote_path=remote_path)
    for rel in extra_files:
        image = image.add_local_file(rel, remote_path=str(remote / rel))

    # 3. Optional (mount-if-on-disk).
    for rel in optional_dirs:
        local = root / rel
        if local.is_dir():
            image = image.add_local_dir(rel, remote_path=str(remote / rel))
    for rel in optional_files:
        local = root / rel
        if local.is_file():
            image = image.add_local_file(rel, remote_path=str(remote / rel))

    # 4. Trainer-introspected required-input-files + extra-mount-paths.
    if trainer_module_path is not None:
        trainer_module = _import_trainer_module(trainer_module_path)
        missing_required: list[tuple[str, Path]] = []
        for flag, default_path in collect_tier_required_input_files(trainer_module):
            local = _resolve_repo_path(default_path, repo_root=root)
            if not local.is_file():
                missing_required.append((flag, local))
                continue
            try:
                rel = str(local.resolve().relative_to(root))
            except ValueError:
                # Outside repo root; mount under absolute path.
                rel = str(local)
            image = image.add_local_file(rel, remote_path=str(remote / rel))
        if missing_required:
            details = "; ".join(
                f"{flag} default={path}" for flag, path in missing_required
            )
            raise MountManifestError(
                "trainer-declared required_input_file defaults missing on disk: "
                f"{details}. Either generate the missing files or pass an "
                "override path via the corresponding env var / CLI flag."
            )

        missing_extras: list[Path] = []
        for extra in collect_extra_mount_paths(trainer_module):
            local = _resolve_repo_path(extra, repo_root=root)
            if not local.exists():
                missing_extras.append(local)
                continue
            try:
                rel = str(local.resolve().relative_to(root))
            except ValueError:
                rel = str(local)
            if local.is_dir():
                image = image.add_local_dir(rel, remote_path=str(remote / rel))
            else:
                image = image.add_local_file(rel, remote_path=str(remote / rel))
        if missing_extras:
            details = "; ".join(str(p) for p in missing_extras)
            raise MountManifestError(
                "trainer-declared TIER_1_EXTRA_MOUNT_PATHS / MODAL_EXTRA_MOUNT_PATHS "
                f"entries missing on disk: {details}."
            )

    return image


__all__ = [
    "DEFAULT_REMOTE_REPO",
    "STRUCTURAL_MINIMUM_DIRS",
    "STRUCTURAL_MINIMUM_FILES",
    "MountManifestError",
    "build_training_image",
    "collect_extra_mount_paths",
    "collect_tier_required_input_files",
]
