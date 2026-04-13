"""Provider-agnostic tac package bootstrap.

Modeled after the stdlib ``ensurepip`` module: calling :func:`bootstrap` with
no arguments is an idempotent "ensure tac is available" idiom, safe to call at
the top of any cloud script.

Architecture — two phases
--------------------------
**Pre-install phase** (stdlib only, tac not yet available):
    :data:`BOOTSTRAP_STUB` is a multi-line string constant that can be embedded
    verbatim in any Kaggle kernel or experiment script. It is self-contained
    (stdlib-only), handles Kaggle / Modal / Lightning / local path detection,
    prefers uv and falls back to pip, and calls :func:`bootstrap` for
    post-install verification once tac is on the path.

**Post-install phase** (tac available, full verification):
    :func:`bootstrap` verifies the installed wheel has the required modules and
    entrypoints. Called from launchers and from the second stage of
    :data:`BOOTSTRAP_STUB`.

Public API
----------
- :data:`WHEEL_GLOBS` — tuple of glob patterns in priority order
- :data:`DEFAULT_DATASET_HINT` — default dataset slug containing the wheel
- :func:`find_wheel` — provider-agnostic wheel search
- :func:`bootstrap` — idempotent install + verify (the main entry point)
- ``ensure_tac_importable = bootstrap`` — backward-compat alias

Usage
-----
From within a tac module (post-install)::

    from tac.deploy.cloud_bootstrap import bootstrap
    bootstrap()  # no-op if already current

In a generated Kaggle kernel preamble::

    from tac.deploy.cloud_bootstrap import BOOTSTRAP_STUB
    preamble = (
        "from pathlib import Path\\n"
        "SCRIPT_PATH = Path(__file__).resolve()\\n"
        + BOOTSTRAP_STUB
        + "_tac_bootstrap(dataset_hint='comma-lab-private-assets',\\n"
        "                extra_search_dirs=(str(SCRIPT_PATH.parent),))\\n"
    )
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Glob patterns for tac wheel files, in priority order.
#: ``tac-*.whl`` takes precedence over the legacy ``comma_video_lab_ball_pack``
#: name.  Within each pattern the highest-sorted candidate wins (latest version).
WHEEL_GLOBS: tuple[str, ...] = ("tac-*.whl", "comma_video_lab_ball_pack-*.whl")

#: Default dataset slug expected to contain the tac wheel + training assets.
DEFAULT_DATASET_HINT: str = "comma-lab-private-assets"

# ---------------------------------------------------------------------------
# BOOTSTRAP_STUB — embeddable stdlib-only two-stage bootstrap
# ---------------------------------------------------------------------------

#: Self-contained bootstrap function for embedding in Kaggle kernel scripts.
#:
#: Defines ``_tac_bootstrap(**kwargs)`` which callers invoke immediately after
#: embedding.  Requires ``SCRIPT_PATH = Path(__file__).resolve()`` to be
#: defined in the surrounding scope before the call.
#:
#: Stage 1: stdlib-only pre-install (finds wheel, installs via uv or pip).
#: Stage 2: delegates to :func:`bootstrap` for full post-install verification.
#:
#: Parameters accepted by ``_tac_bootstrap``:
#:   - ``input_root``: explicit dataset mount root (Path or None for auto-detect)
#:   - ``dataset_hint``: dataset slug subdirectory (default "comma-lab-private-assets")
#:   - ``verify_submodule``: dotted module to verify after install (e.g.
#:     "tac.deploy.kaggle.runner")
#:   - ``extra_search_dirs``: additional dirs to search for the wheel (e.g.
#:     ``(str(SCRIPT_PATH.parent),)`` to check the kernel bundle dir first)
BOOTSTRAP_STUB: str = '''\
def _tac_bootstrap(
    *,
    input_root=None,
    dataset_hint="comma-lab-private-assets",
    verify_submodule=None,
    extra_search_dirs=(),
):
    """Two-stage tac bootstrap: stdlib-only pre-install + cloud_bootstrap verify."""
    import os as _os, shutil as _shutil, subprocess as _sp, sys as _sys
    from pathlib import Path as _Path

    # Stage 1 — idempotency: skip if tac (and optional submodule) already importable
    def _imp(m):
        try: __import__(m); return True
        except ImportError: return False

    if _imp("tac") and (verify_submodule is None or _imp(verify_submodule)):
        return

    # Stage 1 — resolve input root without cloud_paths (stdlib only)
    if input_root is None:
        _env = _os.environ.get("CLOUD_INPUT_ROOT", "").strip()
        if _env:
            input_root = _Path(_env)
        elif _os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            input_root = _Path("/kaggle/input")
        elif _os.environ.get("MODAL_IS_CONTAINER", "").lower() in ("1", "true", "yes"):
            input_root = _Path("/vol/input")
        elif _Path("/teamspace").exists():
            input_root = _Path("/teamspace/input")
        else:
            input_root = _Path.cwd() / "data" / "input"

    # Stage 1 — find wheel: extra dirs > dataset subdir > full input tree
    _search = [
        *[_Path(d) for d in extra_search_dirs if d],
        input_root / dataset_hint if (input_root and (input_root / dataset_hint).exists()) else None,
        input_root,
    ]
    _wheel = None
    for _root in _search:
        if _root is None or not _root.exists():
            continue
        for _pat in ("tac-*.whl", "comma_video_lab_ball_pack-*.whl"):
            _hits = sorted(_root.rglob(_pat))
            if _hits:
                _wheel = _hits[-1]
                break
        if _wheel:
            break
    if _wheel is None:
        raise ImportError(
            f"tac wheel not found.\\n"
            f"  Searched: {[str(r) for r in _search if r is not None]}\\n"
            f"  Upload tac-*.whl to the {dataset_hint!r} dataset:\\n"
            f"    kaggle datasets version -p dist/ -m \\'tac vX.Y.Z\\'"
        )

    # Stage 1 — install: uv preferred, pip fallback
    _uv = _shutil.which("uv") or next(
        (str(c) for c in (
            _Path.home() / ".local" / "bin" / "uv",
            _Path.home() / ".cargo" / "bin" / "uv",
            _Path("/usr/local/bin/uv"),
            _Path("/opt/conda/bin/uv"),
        ) if c.exists()),
        None,
    )
    if _uv:
        _sp.check_call([_uv, "pip", "install", "--system", "-q", "--no-deps", str(_wheel)])
    else:
        _sp.check_call([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(_wheel)])

    # Stage 2 — full post-install verification via cloud_bootstrap
    from tac.deploy.cloud_bootstrap import bootstrap as _cb
    _cb(
        input_root=input_root,
        dataset_hint=dataset_hint,
        verify_submodule=verify_submodule,
        extra_roots=tuple(_Path(d) for d in extra_search_dirs if d),
    )
'''


# ---------------------------------------------------------------------------
# Internal helpers (no tac imports — safe to call immediately post-install)
# ---------------------------------------------------------------------------

def _fallback_input_root() -> Path:
    """Best-effort input root without importing cloud_paths."""
    if os.environ.get("CLOUD_INPUT_ROOT"):
        return Path(os.environ["CLOUD_INPUT_ROOT"])
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/input")
    if os.environ.get("MODAL_IS_CONTAINER", "").lower() in ("1", "true", "yes"):
        return Path("/vol/input")
    if Path("/teamspace").exists():
        return Path("/teamspace/input")
    return Path.cwd() / "data" / "input"


def _try_uv() -> str | None:
    """Return path to uv binary, or None if not found. Never raises."""
    uv_override = os.environ.get("CLOUD_UV_PATH", "").strip()
    if uv_override and Path(uv_override).exists():
        return uv_override
    existing = shutil.which("uv")
    if existing:
        return existing
    for c in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
        Path("/opt/conda/bin/uv"),
    ):
        if c.exists():
            return str(c)
    return None


def _install_wheel(wheel: Path) -> None:
    """Install a wheel file via uv (preferred) or pip (fallback)."""
    uv = _try_uv()
    if uv:
        subprocess.check_call(
            [uv, "pip", "install", "--system", "-q", "--no-deps", str(wheel)]
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel)]
        )


def _is_importable(dotted: str) -> bool:
    """Return True if the dotted module is importable right now."""
    try:
        __import__(dotted)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_wheel(
    input_root: Path | None = None,
    *,
    dataset_hint: str = DEFAULT_DATASET_HINT,
    globs: tuple[str, ...] = WHEEL_GLOBS,
    extra_roots: tuple[Path, ...] = (),
) -> Path:
    """Find the latest tac wheel in *input_root*.

    Search order:
      1. *extra_roots* (caller-provided, e.g. the kernel bundle script dir)
      2. ``input_root / dataset_hint`` (exact dataset mount subdirectory)
      3. ``input_root`` (full rglob fallback)

    If *input_root* is ``None``, auto-detects the platform via
    :class:`tac.deploy.cloud_paths.CloudPaths` when tac is already importable,
    or falls back to env-variable heuristics otherwise.

    Args:
        input_root:   root of the cloud dataset mount. ``None`` → auto-detect.
        dataset_hint: dataset slug subdirectory to search first.
        globs:        glob patterns tried in priority order; highest sort wins
                      within each pattern.
        extra_roots:  additional directories to search before *input_root*.

    Returns:
        :class:`~pathlib.Path` to the latest matching wheel file.

    Raises:
        ImportError: with actionable upload instructions if no wheel is found.
    """
    if input_root is None:
        try:
            from tac.deploy.cloud_paths import CloudPaths
            input_root = CloudPaths.detect().input_root
        except ImportError:
            input_root = _fallback_input_root()

    search_roots: list[Path] = [
        *extra_roots,
        input_root / dataset_hint,
        input_root,
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in globs:
            candidates = sorted(root.rglob(pattern))
            if candidates:
                return candidates[-1]

    raise ImportError(
        f"tac wheel not found.\n"
        f"  Searched: {search_roots}\n"
        f"  Patterns: {globs}\n"
        f"  Upload with: kaggle datasets version -p dist/ -m 'tac vX.Y.Z'\n"
        f"  to the {dataset_hint!r} dataset."
    )


def bootstrap(
    input_root: Path | None = None,
    *,
    dataset_hint: str = DEFAULT_DATASET_HINT,
    verify_module: str = "tac",
    verify_submodule: str | None = None,
    entrypoint_symbols: tuple[str, ...] = (),
    extra_roots: tuple[Path, ...] = (),
) -> None:
    """Ensure tac is installed and the specified modules are importable.

    Idempotent — safe to call at the top of any cloud script.  Returns
    immediately if tac (and all requested verify targets) are already
    importable.  Otherwise locates the wheel, installs it (uv preferred,
    pip fallback), and verifies.

    Modeled after :func:`ensurepip.bootstrap`: calling ``bootstrap()`` with
    no arguments is the canonical "ensure tac is available" idiom.

    Args:
        input_root:         dataset mount root.  ``None`` → auto-detect.
        dataset_hint:       dataset slug subdirectory containing the wheel.
        verify_module:      top-level module to confirm importable after install.
        verify_submodule:   additional dotted module to verify (e.g.
                            ``"tac.deploy.kaggle.runner"`` for runners that need
                            the deploy subpackages added in tac v1.0.0).
        entrypoint_symbols: names to verify as attributes of ``tac.entrypoints``
                            after install (used by generated kernel preambles).
        extra_roots:        extra wheel search paths in addition to *input_root*.

    Raises:
        ImportError: if the wheel cannot be found or post-install verification
                     fails, with actionable instructions for re-uploading.
    """
    # Phase 1 — idempotency
    if _is_importable(verify_module):
        if verify_submodule is None or _is_importable(verify_submodule):
            if not entrypoint_symbols:
                return
            # Also check entrypoint symbols before short-circuiting
            try:
                from tac import entrypoints as _ep
                missing = [s for s in entrypoint_symbols if not hasattr(_ep, s)]
                if not missing:
                    return
            except ImportError:
                pass  # fall through to install

    # Phase 2 — find + install
    wheel = find_wheel(input_root, dataset_hint=dataset_hint, extra_roots=extra_roots)
    print(f"  [cloud_bootstrap] Installing {wheel.name} ...")
    _install_wheel(wheel)

    # Phase 3 — post-install verification
    if not _is_importable(verify_module):
        raise ImportError(
            f"Installed {wheel.name} but {verify_module!r} is not importable.\n"
            f"  The wheel may be outdated or built without this module.\n"
            f"  Rebuild and re-upload:\n"
            f"    uv build && kaggle datasets version -p dist/ -m 'tac vX.Y.Z'"
        )

    if verify_submodule and not _is_importable(verify_submodule):
        raise ImportError(
            f"Installed {wheel.name} but {verify_submodule!r} is not importable.\n"
            f"  This wheel may be pre-v1.0.0 and missing the deploy subpackages.\n"
            f"  Upload tac-1.0.0+ to the {dataset_hint!r} dataset:\n"
            f"    kaggle datasets version -p dist/ -m 'tac v1.0.0'\n"
            f"  (Found wheel at: {wheel})"
        )

    if entrypoint_symbols:
        try:
            from tac import entrypoints as _ep
            missing = [s for s in entrypoint_symbols if not hasattr(_ep, s)]
            if missing:
                raise ImportError(
                    f"Installed {wheel.name} but tac.entrypoints is missing: {missing}.\n"
                    f"  Rebuild and re-upload:\n"
                    f"    uv build && kaggle datasets version -p dist/ -m 'tac vX.Y.Z'"
                )
        except ImportError as exc:
            if "entrypoints is missing" in str(exc):
                raise
            raise ImportError(
                f"Installed {wheel.name} but tac.entrypoints is not importable.\n"
                f"  Rebuild and re-upload:\n"
                f"    uv build && kaggle datasets version -p dist/ -m 'tac vX.Y.Z'"
            ) from exc


# Backward-compatibility alias — used in bootstrap_codegen-generated preambles
# and in legacy code that imported ensure_tac_importable directly.
ensure_tac_importable = bootstrap
