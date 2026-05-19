# SPDX-License-Identifier: MIT
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

import hashlib
import importlib
import importlib.util
import os
import stat
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REMOTE_REPO = "/workspace/pact"
DEFAULT_RESULTS_IGNORE: tuple[str, ...] = ("results/**",)

# ----------------------------------------------------------------------------
# Catalog #165 (FIX-I, D2): Modal upload-race policy via mtime-stability check.
#
# Bug-class anchor: WWW4 (2026-05-12) surfaced that Modal v2/v3 dispatches
# failed because FIX-D's concurrent writes to ``src/tac/composition/registry.py``
# (46257B -> 52539B over 30s) hit Modal's ``add_local_dir`` mid-upload. Modal
# captures a partial file snapshot of an actively-being-written source tree,
# the GPU container runs the partial code, and the dispatch crashes or silently
# corrupts. The bug is structural: ``add_local_dir`` walks the filesystem at
# upload time and any file the operator (or a sibling subagent) is currently
# editing produces a torn read.
#
# Fix design (D2a in the WWW4 enumeration): before ``add_local_*`` calls fire,
# hash the metadata/content fingerprint of every file in the mount set, sleep N
# seconds, recompute. If unchanged, proceed. If changed (=> someone is writing
# inside our window), retry up to MAX_RETRIES. After exhaustion fail-closed with
# a ``MountUploadRaceError`` so the operator can pause and rerun.
#
# Same-line ``# MODAL_UPLOAD_RACE_OK:<reason>`` waiver is honored at the
# preflight Catalog #165 layer (for callsites that bypass the canonical
# builder); the in-process knob is the ``TAC_MODAL_MTIME_STABILITY_DISABLED``
# env var (tests + emergency operator override only).
#
# Sister gates:
#   * Catalog #153 (check_modal_dispatcher_uses_canonical_mount_builder) —
#     ensures the canonical builder is used at all.
#   * Catalog #165 (check_modal_mount_builder_uses_mtime_stability_check) —
#     ensures the canonical builder honors the mtime-stability invariant.
# Together they close the upload-race bug class at both the wire-up layer
# and the policy layer.
# ----------------------------------------------------------------------------
DEFAULT_MTIME_STABILITY_WINDOW_SECONDS: float = 2.0
DEFAULT_MTIME_STABILITY_MAX_RETRIES: int = 3
DEFAULT_MTIME_STABILIZATION_ATTEMPTS: int = 3
DEFAULT_MTIME_STABILIZATION_COOLDOWN_SECONDS: float = 1.0
DEFAULT_MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS: int = 3
DEFAULT_MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_SECONDS: float = 5.0
MOUNT_FINGERPRINT_CONTENT_CHUNK_BYTES: int = 1024 * 1024
_MTIME_STABILITY_DISABLE_ENV = "TAC_MODAL_MTIME_STABILITY_DISABLED"
_MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS_ENV = "TAC_MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS"
_MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_ENV = (
    "TAC_MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_SECONDS"
)

MODAL_MOUNT_UPLOAD_RACE_MARKERS: tuple[str, ...] = (
    "mountuploadraceerror",
    "fingerprint is unstable",
    "modified during build process",
    "was modified during build",
    "files being uploaded were modified",
)
MODAL_DISPATCH_SPAWN_MARKERS: tuple[str, ...] = (
    "[modal_train_lane] dispatch_completed call_id=fc-",
    "call_id=fc-",
    "dispatched via .spawn()",
)

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


class MountUploadRaceError(MountManifestError):
    """Raised when the mount set's metadata/content fingerprint changes during
    the stability window, indicating a sibling writer is racing the Modal
    upload.

    Catalog #165 (FIX-I, D2). The fix-class is "fail-closed and let the
    operator pause"; silently retrying forever masks runaway writers.
    """


def modal_output_indicates_mount_upload_race(output: str) -> bool:
    """Return True when Modal/local output names the mount-upload race class."""

    lowered = output.lower()
    return any(marker in lowered for marker in MODAL_MOUNT_UPLOAD_RACE_MARKERS)


def modal_output_indicates_spawned_call(output: str) -> bool:
    """Return True once output proves a Modal function call was spawned.

    Operator-side retry wrappers must not retry after this point; a retry could
    create duplicate paid work. Mount-upload retries are pre-spawn only.
    """

    lowered = output.lower()
    return any(marker in lowered for marker in MODAL_DISPATCH_SPAWN_MARKERS)


def _positive_int_from_env(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _nonnegative_float_from_env(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, value)


def modal_mount_upload_cli_retry_settings(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, float]:
    """Return bounded retry settings for pre-spawn Modal upload races.

    This does not disable Catalog #165. It only lets operator wrappers retry a
    Modal CLI invocation that failed before ``.spawn()`` because Modal or the
    local guard observed source mtimes changing during upload.
    """

    env = os.environ if environ is None else environ
    return (
        _positive_int_from_env(
            env,
            _MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS_ENV,
            DEFAULT_MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS,
        ),
        _nonnegative_float_from_env(
            env,
            _MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_ENV,
            DEFAULT_MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_SECONDS,
        ),
    )


def _hash_path_fingerprint_entry(
    hasher: Any,
    path: Path,
    st: os.stat_result,
) -> None:
    """Hash one filesystem entry using metadata plus streamed content.

    The content pass is intentionally local to the Modal mount-stability
    surface. It reads regular files in bounded chunks so same-size rewrites with
    restored ``st_mtime_ns`` are still detected without materializing large
    files in memory.
    """

    hasher.update(
        f"{path!s}\x00{st.st_mode}\x00{st.st_mtime_ns}\x00{st.st_size}\x00".encode()
    )
    mode = stat.S_IFMT(st.st_mode)
    if stat.S_ISREG(mode):
        hasher.update(b"\x00CONTENT_BEGIN\x00")
        try:
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(MOUNT_FINGERPRINT_CONTENT_CHUNK_BYTES)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except FileNotFoundError:
            hasher.update(b"\x00CONTENT_MISSING\x00")
        except OSError as exc:
            hasher.update(
                f"\x00CONTENT_OSERR\x00{type(exc).__name__}\x00{exc.errno}\x00".encode()
            )
        hasher.update(b"\x00CONTENT_END\x00")
    elif stat.S_ISLNK(mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            target = f"\x00READLINK_OSERR\x00{type(exc).__name__}\x00{exc.errno}"
        hasher.update(f"\x00SYMLINK\x00{target}\x00".encode())


MountFingerprintEntry = Path | tuple[Path, tuple[str, ...] | None]


def _normalize_fingerprint_entry(
    entry: MountFingerprintEntry,
) -> tuple[Path, tuple[str, ...] | None]:
    if isinstance(entry, tuple):
        path, ignore = entry
        return path, ignore
    return entry, None


def _modal_ignore_matcher(patterns: tuple[str, ...] | None) -> Any | None:
    """Build the same pattern matcher Modal uses for ``Image.add_local_dir``."""

    if not patterns:
        return None

    from modal.file_pattern_matcher import FilePatternMatcher

    return FilePatternMatcher(*patterns)


def _normalized_descendant_prune_prefixes(
    patterns: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return directory prefixes where ``dir/**`` ignores every descendant.

    Modal receives patterns such as ``results/**``. The stability guard must
    honor the same uploaded-file set. Modal's matcher is authoritative for
    file inclusion; this helper only adds safe early pruning for plain
    descendant-only patterns so the guard does not walk/read large result
    trees that Modal will never upload.
    """

    if not patterns:
        return ()
    prefixes: list[str] = []
    for pattern in patterns:
        normalized = pattern.strip()
        if not normalized or normalized.startswith("!"):
            continue
        normalized = os.path.normpath(normalized.strip(os.path.sep)).replace(os.path.sep, "/")
        if normalized.endswith("/**"):
            prefix = normalized[:-3].rstrip("/")
            if prefix and prefix not in prefixes:
                prefixes.append(prefix)
    return tuple(prefixes)


def _can_prune_modal_ignored_dir(
    rel: Path,
    patterns: tuple[str, ...] | None,
    matcher: Any | None,
) -> bool:
    """Return whether the guard may skip recursing into ``rel`` entirely."""

    if matcher is None:
        return False
    can_prune = getattr(matcher, "can_prune_directories", lambda: False)
    if not can_prune():
        return False
    rel_text = rel.as_posix()
    for prefix in _normalized_descendant_prune_prefixes(patterns):
        if rel_text == prefix:
            return True
    return bool(matcher(rel))


def _iter_modal_upload_fingerprint_paths(
    root: Path,
    patterns: tuple[str, ...] | None,
) -> Iterable[Path]:
    """Yield local paths that affect Modal's upload set for one mounted dir."""

    matcher = _modal_ignore_matcher(patterns)
    for current, dir_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        dir_names.sort()
        file_names.sort()
        if matcher is not None:
            dir_names[:] = [
                name
                for name in dir_names
                if not _can_prune_modal_ignored_dir(
                    (current_path / name).relative_to(root),
                    patterns,
                    matcher,
                )
            ]
        for name in file_names:
            local_path = current_path / name
            rel_path = local_path.relative_to(root)
            if matcher is not None and matcher(rel_path):
                continue
            yield local_path


def _hash_mount_set_fingerprint(
    paths: Iterable[MountFingerprintEntry],
) -> tuple[str, list[Path]]:
    """Return ``(sha256_hex, missing_paths)`` for every path in ``paths``.

    Directories are walked recursively. Symlinks are stat'd, not followed,
    because Modal copies the symlink contents (not target) by default. Missing
    paths are returned in ``missing_paths`` but do NOT short-circuit the hash
    (they are absorbed into the hash via a sentinel marker so callers can see
    "the missing-path set changed too" as instability).
    """

    hasher = hashlib.sha256()
    missing: list[Path] = []
    # Deterministic ordering: caller's order is preserved, then each
    # directory walked in sorted order.
    for entry in paths:
        p, ignore = _normalize_fingerprint_entry(entry)
        try:
            st = p.lstat()
        except FileNotFoundError:
            hasher.update(f"\x00MISSING\x00{p!s}\x00".encode())
            missing.append(p)
            continue
        except OSError:
            hasher.update(f"\x00OSERR\x00{p!s}\x00".encode())
            continue
        if p.is_dir() and not p.is_symlink():
            # Walk the same uploaded-file set Modal sees, with safe pruning
            # for ignored result trees.
            for sub in _iter_modal_upload_fingerprint_paths(p, ignore):
                try:
                    sub_st = sub.lstat()
                except FileNotFoundError:
                    hasher.update(f"\x00MISSING\x00{sub!s}\x00".encode())
                    continue
                except OSError:
                    continue
                _hash_path_fingerprint_entry(hasher, sub, sub_st)
        else:
            _hash_path_fingerprint_entry(hasher, p, st)
    return hasher.hexdigest(), missing


def verify_mount_set_mtime_stability(
    paths: Iterable[MountFingerprintEntry],
    *,
    window_seconds: float = DEFAULT_MTIME_STABILITY_WINDOW_SECONDS,
    max_retries: int = DEFAULT_MTIME_STABILITY_MAX_RETRIES,
    sleep_fn: Any = time.sleep,
) -> None:
    """Verify the metadata/content fingerprint of ``paths`` is stable for
    ``window_seconds``.

    Procedure:
      1. Hash the fingerprint.
      2. Sleep ``window_seconds``.
      3. Recompute the fingerprint.
      4. If unchanged -> return.
      5. If changed -> retry up to ``max_retries`` times.
      6. If still changing after ``max_retries`` -> raise
         ``MountUploadRaceError`` with a diagnostic message identifying the
         instability window and retry count.

    The opt-out env var ``TAC_MODAL_MTIME_STABILITY_DISABLED=1`` skips the
    check entirely; use only for tests / emergency operator override (and
    leave a note in the dispatch log).

    Catalog #165. Sister of Catalog #153.
    """

    if os.environ.get(_MTIME_STABILITY_DISABLE_ENV, "").strip() == "1":
        return

    path_list = list(paths)
    if not path_list:
        return

    for _attempt in range(1, max_retries + 1):
        fp_before, _ = _hash_mount_set_fingerprint(path_list)
        sleep_fn(window_seconds)
        fp_after, _ = _hash_mount_set_fingerprint(path_list)
        if fp_before == fp_after:
            return

    raise MountUploadRaceError(
        "Modal mount set metadata/content fingerprint is unstable after "
        f"{max_retries} retries (window={window_seconds}s each). A sibling "
        "writer is racing the Modal upload. Pause concurrent writers and "
        "re-run, or set TAC_MODAL_MTIME_STABILITY_DISABLED=1 to bypass "
        "(NOT recommended — bytes Modal uploads will be torn). Catalog #165 "
        "(FIX-I, D2)."
    )


def stabilize_mount_set_for_modal_upload(
    paths: Iterable[MountFingerprintEntry],
    *,
    window_seconds: float = DEFAULT_MTIME_STABILITY_WINDOW_SECONDS,
    max_attempts: int = DEFAULT_MTIME_STABILIZATION_ATTEMPTS,
    max_retries_per_attempt: int = DEFAULT_MTIME_STABILITY_MAX_RETRIES,
    cooldown_seconds: float = DEFAULT_MTIME_STABILIZATION_COOLDOWN_SECONDS,
    sleep_fn: Any = time.sleep,
) -> int:
    """Wait until the Modal mount set is stable, or fail closed.

    ``verify_mount_set_mtime_stability`` is the single-attempt guard. This
    helper is the canonical bounded stabilizer for operator paths where sibling
    subagents may finish writing seconds later. It never bypasses the guard: it
    repeats the guard, with an optional cooldown, and raises
    ``MountUploadRaceError`` after ``max_attempts``.

    Returns the 1-based attempt number that first observed a stable mount set.
    """

    path_list = list(paths)
    if not path_list:
        return 0
    attempts = max(1, int(max_attempts))
    retries_per_attempt = max(1, int(max_retries_per_attempt))
    last_error: MountUploadRaceError | None = None
    for attempt in range(1, attempts + 1):
        try:
            verify_mount_set_mtime_stability(
                path_list,
                window_seconds=window_seconds,
                max_retries=retries_per_attempt,
                sleep_fn=sleep_fn,
            )
            return attempt
        except MountUploadRaceError as exc:
            last_error = exc
            if attempt < attempts and cooldown_seconds > 0:
                sleep_fn(cooldown_seconds)

    raise MountUploadRaceError(
        "Modal mount set did not stabilize after "
        f"{attempts} attempt(s) "
        f"(window={window_seconds}s, retries_per_attempt={retries_per_attempt}, "
        f"cooldown={cooldown_seconds}s). Last guard failure: {last_error}. "
        "Catalog #165 remains fail-closed to avoid torn Modal uploads."
    ) from last_error


def _collect_paths_for_stability_check(
    *,
    root: Path,
    skip_structural: bool,
    extra_dirs: Iterable[tuple[str, tuple[str, ...] | None]],
    extra_files: Iterable[str],
    optional_dirs: Iterable[str],
    optional_files: Iterable[str],
    trainer_required_files: Iterable[Path] = (),
    trainer_extra_paths: Iterable[Path] = (),
) -> list[MountFingerprintEntry]:
    """Return the unioned list of local on-disk paths that ``build_training_image``
    will hand to Modal for upload. Used by ``verify_mount_set_mtime_stability``.
    """

    paths: list[MountFingerprintEntry] = []
    if not skip_structural:
        for rel, ignore in STRUCTURAL_MINIMUM_DIRS:
            paths.append((root / rel, ignore))
        for rel in STRUCTURAL_MINIMUM_FILES:
            paths.append(root / rel)
    for rel, ignore in extra_dirs:
        paths.append((root / rel, ignore))
    for rel in extra_files:
        paths.append(root / rel)
    for rel in optional_dirs:
        p = root / rel
        if p.is_dir():
            paths.append(p)
    for rel in optional_files:
        p = root / rel
        if p.is_file():
            paths.append(p)
    for default_path in trainer_required_files:
        paths.append(_resolve_repo_path(default_path, repo_root=root))
    for extra in trainer_extra_paths:
        paths.append(_resolve_repo_path(extra, repo_root=root))
    return paths


def _resolve_repo_path(rel_or_abs: str | Path, *, repo_root: Path) -> Path:
    """Resolve a path string relative to repo root unless already absolute."""

    p = Path(rel_or_abs)
    return p if p.is_absolute() else (repo_root / p)


def _trainer_module_source_path(
    trainer_module_path: str | Path,
    *,
    repo_root: Path,
) -> Path | None:
    """Best-effort source path for pre-import stability checks.

    ``importlib`` executes trainer top-level code. When a sibling writer is
    editing trainer source, importing before the stability guard can observe a
    torn module. This helper resolves common file-path and repo-local dotted
    module forms without importing anything.
    """

    path_str = str(trainer_module_path)
    if path_str.endswith(".py") or "/" in path_str or os.sep in path_str:
        return _resolve_repo_path(path_str, repo_root=repo_root)

    module_rel = Path(*path_str.split("."))
    file_candidate = repo_root / module_rel.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    package_candidate = repo_root / module_rel / "__init__.py"
    if package_candidate.is_file():
        return package_candidate
    return None


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
        # Register module in sys.modules BEFORE exec so @dataclass's _is_type
        # lookup (which calls sys.modules.get(cls.__module__).__dict__) works
        # for trainers that define module-level dataclasses. Without this,
        # @dataclass on any module-level class raises AttributeError because
        # sys.modules.get(...) returns None. Cleanup after exec so we don't
        # pollute sys.modules with the synthetic introspect name.
        synthetic_name = spec.name
        import sys as _sys
        _sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - surfaced at build time only
            _sys.modules.pop(synthetic_name, None)
            raise MountManifestError(
                f"trainer module {file_path} raised at import time: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            # Defensive cleanup: only remove if we still own the slot (no nested
            # spec.loader.exec_module may have replaced it via a sister import).
            if _sys.modules.get(synthetic_name) is module:
                _sys.modules.pop(synthetic_name, None)
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
    mtime_stability_check: bool = True,
    mtime_stability_window_seconds: float = DEFAULT_MTIME_STABILITY_WINDOW_SECONDS,
    mtime_stability_max_retries: int = DEFAULT_MTIME_STABILITY_MAX_RETRIES,
    mtime_stability_stabilization_attempts: int = DEFAULT_MTIME_STABILIZATION_ATTEMPTS,
    mtime_stability_retry_cooldown_seconds: float = DEFAULT_MTIME_STABILIZATION_COOLDOWN_SECONDS,
    mtime_stability_sleep_fn: Any = time.sleep,
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

    # 0. Catalog #165 (FIX-I, D2): mtime-stability check across the unioned
    # mount set BEFORE any ``add_local_*`` call. Sibling subagents writing
    # to files Modal is about to upload would otherwise produce torn reads.
    #
    # When a trainer module is provided, first stabilize the import surface
    # that can be resolved without importing. Importing top-level trainer code
    # before this guard would let Python observe a torn source file. After the
    # import, run the full mount-set guard including trainer-declared required
    # inputs and extra paths before any Modal ``add_local_*`` call.
    _trainer_module_cached: Any = None
    _trainer_required_files_cached: list[tuple[str, Path]] = []
    _trainer_extra_paths_cached: list[Path] = []
    if mtime_stability_check and trainer_module_path is not None:
        pre_import_paths = _collect_paths_for_stability_check(
            root=root,
            skip_structural=skip_structural,
            extra_dirs=extra_dirs,
            extra_files=extra_files,
            optional_dirs=optional_dirs,
            optional_files=optional_files,
        )
        trainer_source_path = _trainer_module_source_path(
            trainer_module_path,
            repo_root=root,
        )
        if trainer_source_path is not None:
            pre_import_paths.append(trainer_source_path)
        if mtime_stability_stabilization_attempts <= 1:
            verify_mount_set_mtime_stability(
                pre_import_paths,
                window_seconds=mtime_stability_window_seconds,
                max_retries=mtime_stability_max_retries,
                sleep_fn=mtime_stability_sleep_fn,
            )
        else:
            stabilize_mount_set_for_modal_upload(
                pre_import_paths,
                window_seconds=mtime_stability_window_seconds,
                max_attempts=mtime_stability_stabilization_attempts,
                max_retries_per_attempt=mtime_stability_max_retries,
                cooldown_seconds=mtime_stability_retry_cooldown_seconds,
                sleep_fn=mtime_stability_sleep_fn,
            )

    if trainer_module_path is not None:
        _trainer_module_cached = _import_trainer_module(trainer_module_path)
        _trainer_required_files_cached = collect_tier_required_input_files(
            _trainer_module_cached
        )
        _trainer_extra_paths_cached = collect_extra_mount_paths(_trainer_module_cached)

    if mtime_stability_check:
        all_mount_paths = _collect_paths_for_stability_check(
            root=root,
            skip_structural=skip_structural,
            extra_dirs=extra_dirs,
            extra_files=extra_files,
            optional_dirs=optional_dirs,
            optional_files=optional_files,
            trainer_required_files=[
                path for _flag, path in _trainer_required_files_cached
            ],
            trainer_extra_paths=_trainer_extra_paths_cached,
        )
        if mtime_stability_stabilization_attempts <= 1:
            verify_mount_set_mtime_stability(
                all_mount_paths,
                window_seconds=mtime_stability_window_seconds,
                max_retries=mtime_stability_max_retries,
                sleep_fn=mtime_stability_sleep_fn,
            )
        else:
            stabilize_mount_set_for_modal_upload(
                all_mount_paths,
                window_seconds=mtime_stability_window_seconds,
                max_attempts=mtime_stability_stabilization_attempts,
                max_retries_per_attempt=mtime_stability_max_retries,
                cooldown_seconds=mtime_stability_retry_cooldown_seconds,
                sleep_fn=mtime_stability_sleep_fn,
            )

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
        # Use the trainer module + lists pre-cached above (step 0) so we do
        # not double-import.
        missing_required: list[tuple[str, Path]] = []
        for flag, default_path in _trainer_required_files_cached:
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
        for extra in _trainer_extra_paths_cached:
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
    "DEFAULT_MODAL_MOUNT_UPLOAD_CLI_MAX_ATTEMPTS",
    "DEFAULT_MODAL_MOUNT_UPLOAD_CLI_RETRY_SLEEP_SECONDS",
    "DEFAULT_MTIME_STABILITY_MAX_RETRIES",
    "DEFAULT_MTIME_STABILITY_WINDOW_SECONDS",
    "DEFAULT_MTIME_STABILIZATION_ATTEMPTS",
    "DEFAULT_MTIME_STABILIZATION_COOLDOWN_SECONDS",
    "DEFAULT_REMOTE_REPO",
    "MODAL_DISPATCH_SPAWN_MARKERS",
    "MODAL_MOUNT_UPLOAD_RACE_MARKERS",
    "STRUCTURAL_MINIMUM_DIRS",
    "STRUCTURAL_MINIMUM_FILES",
    "MountManifestError",
    "MountUploadRaceError",
    "build_training_image",
    "collect_extra_mount_paths",
    "collect_tier_required_input_files",
    "modal_mount_upload_cli_retry_settings",
    "modal_output_indicates_mount_upload_race",
    "modal_output_indicates_spawned_call",
    "stabilize_mount_set_for_modal_upload",
    "verify_mount_set_mtime_stability",
]
