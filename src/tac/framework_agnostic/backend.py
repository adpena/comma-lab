# SPDX-License-Identifier: MIT
"""Canonical backend selection for framework-agnostic primitives.

Sister of ``tac.substrates._shared.inflate_runtime.select_inflate_device`` per
Catalog #205 at the **framework-selection surface at training-time** (the
sister gate covers **device-selection at inflate-time**).

Per operator NON-NEGOTIABLE META directive 2026-05-28 verbatim: *"remmebr MLX
first but agnostic portability via numpy and tinygrad like primitives and
helpers or decorators or whatever"*.

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
directive:
  * TRAINING MLX-first on M5 Max + INFLATE numpy-portable (no MLX dep)
  * INFLATE bridge contract: MLX state_dict → npz → ZIP-member → numpy
    inflate primitives (≤200 LOC + ≤2 ext deps per HNeRV parity L4)
  * Each substrate INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)"
+ "Forbidden score claims" non-negotiables: AUTO selection NEVER silently
falls back to a non-promotable framework when the caller expected a
contest-grade framework. The canonical AUTO order routes through the
canonical helper at the calling site.

Per CLAUDE.md "tac stays clean" non-negotiable: this canonical helper lives
in ``tac.framework_agnostic`` so any reusable primitive can consume it
without importing torch / mlx / tinygrad at module-top (deferred-import
discipline per the 4-backend contract).

Cross-references:
  * Catalog #205 — sister gate at the inflate-time surface
  * Catalog #335 — canonical cathedral consumer contract (sister at
    auto-discovery surface; this helper is the upstream selection primitive
    that consumers can route through)
  * Catalog #357 — Tier A canonical-routing markers (any backend-routing
    consumer's contributions are observability-only per #341)
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * Catalog #340 — sister-checkpoint guard discipline (NEW framework-
    selection surface; runs BEFORE bare ``git add``)
"""
from __future__ import annotations

import enum
import os
import platform
from collections.abc import Sequence


class Backend(enum.Enum):
    """Canonical backend taxonomy for framework-agnostic primitives.

    The ``AUTO`` sentinel is the canonical caller default; resolves at
    call-time via :func:`select_backend` to the first available concrete
    backend per the OS/architecture-aware priority order.

    Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
    directive:
      * Darwin ARM64: MLX → NUMPY (training MLX-first; inflate numpy-portable)
      * Linux CUDA: PYTORCH → NUMPY (contest-resolution paths)
      * Linux CPU: NUMPY (deterministic numpy-portable inflate)
      * Other: NUMPY (canonical fallback; always available)

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA":
      * MLX is non-promotable on its own per Catalog #192/#317; MLX results
        require paired PyTorch CUDA + Linux CPU empirical anchor before any
        contest-grade promotion claim.

    ``TINYGRAD`` is OPTIONAL per Catalog #287; deferred import; selection
    raises a clear error with installation guidance if not installed.
    """

    MLX = "mlx"
    PYTORCH = "pytorch"
    NUMPY = "numpy"
    TINYGRAD = "tinygrad"
    AUTO = "auto"


# Canonical env-var contract. Sister of ``PACT_INFLATE_DEVICE`` per Catalog #205.
DEFAULT_ENV_VAR = "PACT_FRAMEWORK_BACKEND"

# Canonical priority order for AUTO selection per OS/architecture.
# Tested via :func:`select_backend(Backend.AUTO)` + per-platform priority.
_AUTO_PRIORITY_DARWIN_ARM64: tuple[Backend, ...] = (
    Backend.MLX,
    Backend.NUMPY,
)
_AUTO_PRIORITY_LINUX_CUDA: tuple[Backend, ...] = (
    Backend.PYTORCH,
    Backend.NUMPY,
)
_AUTO_PRIORITY_LINUX_CPU: tuple[Backend, ...] = (
    Backend.PYTORCH,
    Backend.NUMPY,
)
_AUTO_PRIORITY_GENERIC: tuple[Backend, ...] = (Backend.NUMPY,)


class BackendUnavailableError(RuntimeError):
    """Raised when a non-AUTO Backend is requested but the framework is not installed.

    Per CLAUDE.md "Comment-only contracts are FORBIDDEN": the error message
    cites the canonical installation route so the operator can resolve
    without re-reading docstrings.

    Sister of ``PACT_INFLATE_DEVICE=cuda but torch.cuda is not available``
    error from ``select_inflate_device`` (Catalog #205).
    """


def _find_spec_safe(name: str) -> bool:
    """Safe deferred-import probe via importlib.util.find_spec.

    Returns True iff the named module can be imported without actually
    importing it. importlib.util must be explicitly imported (not just
    importlib) for find_spec to be available.
    """
    try:
        import importlib.util  # noqa: PLC0415  # deferred-import contract
        spec = importlib.util.find_spec(name)
        return spec is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _is_mlx_available() -> bool:
    """Detect MLX availability via deferred import.

    Returns False on non-Darwin + non-Apple-Silicon hosts without raising.
    MLX is the canonical training framework on M5 Max per CLAUDE.md 8th
    standing directive but not portable.
    """
    return _find_spec_safe("mlx")


def _is_pytorch_available() -> bool:
    """Detect PyTorch availability via deferred import.

    Returns True on Linux + Darwin where torch is the canonical
    contest-resolution framework per Catalog #205 sister discipline.
    """
    return _find_spec_safe("torch")


def _is_numpy_available() -> bool:
    """Detect numpy availability via deferred import.

    numpy is the canonical fallback per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE"
    8th standing directive; should always be available in the canonical
    repo environment.
    """
    return _find_spec_safe("numpy")


def _is_tinygrad_available() -> bool:
    """Detect tinygrad availability via deferred import.

    tinygrad is OPTIONAL per Catalog #287; install via
    ``uv pip install tinygrad`` if needed.
    """
    return _find_spec_safe("tinygrad")


_AVAILABILITY_CHECK = {
    Backend.MLX: _is_mlx_available,
    Backend.PYTORCH: _is_pytorch_available,
    Backend.NUMPY: _is_numpy_available,
    Backend.TINYGRAD: _is_tinygrad_available,
}


def detect_available_backends() -> tuple[Backend, ...]:
    """Return the tuple of concretely-installed backends (excludes AUTO).

    Order: MLX → PYTORCH → NUMPY → TINYGRAD. Caller may filter / re-order
    per per-substrate UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

    Returns:
        Tuple of installed Backend enum members. Always includes Backend.NUMPY
        per canonical repo dependency.
    """
    return tuple(b for b in (Backend.MLX, Backend.PYTORCH, Backend.NUMPY, Backend.TINYGRAD)
                 if _AVAILABILITY_CHECK[b]())


def _platform_priority_order() -> tuple[Backend, ...]:
    """Return canonical AUTO priority for the current platform.

    Darwin ARM64 → MLX-first (per 8th standing directive).
    Linux + torch.cuda → PYTORCH-first (per Catalog #205 sister).
    Other → NUMPY (canonical fallback).
    """
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return _AUTO_PRIORITY_DARWIN_ARM64
    if system == "Linux":
        # Probe CUDA availability via torch (deferred import) — if torch is
        # installed AND torch.cuda.is_available(), prefer PYTORCH path.
        if _is_pytorch_available():
            try:
                import torch  # noqa: PLC0415  # deferred per Catalog #205
                if torch.cuda.is_available():
                    return _AUTO_PRIORITY_LINUX_CUDA
            except (ImportError, RuntimeError):
                pass
        return _AUTO_PRIORITY_LINUX_CPU
    return _AUTO_PRIORITY_GENERIC


def select_backend(
    *,
    override: Backend | None = None,
    env_var: str = DEFAULT_ENV_VAR,
    priority: Sequence[Backend] | None = None,
) -> Backend:
    """Resolve the canonical concrete Backend for the current call site.

    Selection cascade per Catalog #205 sister discipline:

    1. ``override`` kwarg (explicit caller-supplied; highest precedence)
    2. ``env_var`` environment variable (canonical: ``PACT_FRAMEWORK_BACKEND``)
    3. ``priority`` kwarg if supplied; else canonical platform priority
    4. First-available from cascade order

    Args:
        override: Explicit Backend selection; bypasses env-var + priority.
            Pass ``Backend.AUTO`` to defer entirely to env-var / priority.
        env_var: Environment variable name to consult. Default
            ``PACT_FRAMEWORK_BACKEND``. Set ``""`` to skip env-var.
        priority: Optional caller-supplied priority tuple. Defaults to
            canonical platform order via ``_platform_priority_order``.

    Returns:
        A concrete ``Backend`` enum (never ``Backend.AUTO``).

    Raises:
        BackendUnavailableError: If a non-AUTO selection (via override or
            env_var) names a backend that is not installed.
        RuntimeError: If no backends are installed (impossible in canonical
            repo where numpy is a hard dependency).
    """
    # Step 1: explicit override.
    if override is not None and override is not Backend.AUTO:
        if not _AVAILABILITY_CHECK[override]():
            raise BackendUnavailableError(
                f"Backend.{override.name} requested via override= but not "
                f"installed. Install via canonical route: "
                f"{_install_hint(override)}"
            )
        return override

    # Step 2: env var.
    if env_var:
        env_value = os.environ.get(env_var, "").strip().lower()
        if env_value and env_value != "auto":
            try:
                requested = Backend(env_value)
            except ValueError as exc:
                raise BackendUnavailableError(
                    f"{env_var}={env_value!r} unrecognized; expected one of "
                    f"{[b.value for b in Backend if b is not Backend.AUTO]}"
                ) from exc
            if not _AVAILABILITY_CHECK[requested]():
                raise BackendUnavailableError(
                    f"{env_var}={env_value!r} but Backend.{requested.name} is "
                    f"not installed. Install via canonical route: "
                    f"{_install_hint(requested)}"
                )
            return requested

    # Step 3: priority cascade (caller-supplied or canonical platform).
    candidates = tuple(priority) if priority is not None else _platform_priority_order()
    for backend in candidates:
        if backend is Backend.AUTO:
            continue  # skip the sentinel; it should not appear in priority lists
        if _AVAILABILITY_CHECK[backend]():
            return backend

    # Step 4: no concrete backend installed (impossible if numpy hard-dep).
    raise RuntimeError(
        "No framework backends installed. numpy is a hard dependency of "
        "tac per CLAUDE.md 'MLX-FIRST NUMPY-PORTABLE' directive; install "
        "via `uv pip install numpy`."
    )


def _install_hint(backend: Backend) -> str:
    """Return canonical installation instructions for an unavailable backend."""
    if backend is Backend.MLX:
        return "`uv pip install mlx` (Darwin ARM64 / Apple Silicon only)"
    if backend is Backend.PYTORCH:
        return "`uv pip install torch` (CUDA wheel: see CLAUDE.md uv-cu124 trap)"
    if backend is Backend.NUMPY:
        return "`uv pip install numpy` (canonical hard dependency)"
    if backend is Backend.TINYGRAD:
        return "`uv pip install tinygrad` (optional)"
    return "see CLAUDE.md tooling section"


__all__ = [
    "Backend",
    "BackendUnavailableError",
    "DEFAULT_ENV_VAR",
    "detect_available_backends",
    "select_backend",
]
