# SPDX-License-Identifier: MIT
"""Canonical MLX impl of Mamba-2 SSD selective state-space recurrence.

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
directive: training MLX-first on M5 Max + numpy fallback for portability +
inflate numpy-portable bridge. This backend is the PRIMARY substrate-training
surface for sister waves spawned per the 8th directive.

Per operator NON-NEGOTIABLE 2026-05-30 directive verbatim *"wherever we are
missing MLX implementations or any grammar or anything let's do it"*.

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable + Catalog
#192/#317: MLX is non-promotable on its own; MLX results require paired PyTorch
CUDA + Linux CPU empirical anchor before any contest-grade promotion claim.
This backend is the canonical MLX-LOCAL surface for fast candidate generation
+ scorer-response training data + portability engineering + calibrated spend
triage; downstream Modal A100 / Vast.ai 4090 paired empirical anchor is the
contest-axis truth.

Math fidelity vs numpy reference
---------------------------------

Sequential step / forward routines mirror numpy reference math exactly within
MLX float32 numerical tolerance (max_abs < 3e-5 per the canonical drift
discipline per Slot 1303 T3 GRAND COUNCIL MLX-vs-PyTorch drift symposium +
Slot 1255 PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING):

    α_t = mx.exp(dt_t * (-mx.exp(A_log)))         # (B, nheads) → broadcast (B, nheads, 1, 1)
    h_t = α_t · h_{t-1} + dt_t · (B_t ⊗ x_t)      # state update
    y_t = mx.einsum("bhdk,bhk->bhd", h_t, C_t)    # output projection
    y_t += D · x_t                                  # optional skip

Documented adaptations per CLAUDE.md "Forbidden empirical-claim-without-
evidence-tag" + 5-axis taxonomy:

  1. **Axis 4 (math)**: MLX float32 ~3e-5 drift vs numpy float64 is canonical
     per Slot 1303 council verdict; the parity test accepts this band.
  2. **Axis 3 (problem space)**: MLX lazy evaluation — caller may need to
     ``mx.eval(...)`` to materialize tensors for byte-stable comparison.
  3. **Axis 5 (data — MLX Metal kernel)**: this backend uses canonical MLX
     primitives (mx.exp / mx.einsum / broadcast); Metal-kernel optimization
     via mlx.fast.scaled_dot_product_attention adaptations is a future sister
     wave (operator-routable per the optimization standing directive 2026-05-29).

Per CLAUDE.md "Forbidden device-selection defaults": MLX-only path on Darwin
ARM64; on non-Darwin or non-ARM hosts the caller MUST use numpy or PyTorch
backend instead. MLX availability is probed via deferred import per the
canonical pattern in :mod:`tac.framework_agnostic.backend`.

Cross-references:
  * Sister numpy reference: :mod:`.numpy_backend`
  * Sister PyTorch impl: :mod:`.pytorch_backend`
  * Catalog #1297 — Z6PCWM1 byte-stable MLX-vs-PyTorch parity sister
  * Catalog #1255/#1303 — canonical MLX↔PyTorch drift mitigation engineering
  * https://github.com/purohit10saurabh/mamba-ssm-macos — MPS reference adapted
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

__all__ = [
    "Mamba2SSDMLXState",
    "mamba2_ssd_init_state_mlx",
    "mamba2_ssd_step_mlx",
    "mamba2_ssd_forward_sequence_mlx",
    "MLX_AVAILABLE",
]


def _probe_mlx_available() -> bool:
    """Deferred-import MLX probe per Catalog #205 sister discipline."""
    try:
        import importlib.util  # noqa: PLC0415
        return importlib.util.find_spec("mlx") is not None
    except (ImportError, AttributeError, ValueError):
        return False


MLX_AVAILABLE: bool = _probe_mlx_available()


@dataclass(frozen=True)
class Mamba2SSDMLXState:
    """Externalized SSD recurrent state for the MLX backend.

    Shape contract per Dao+Gu 2024 §4 SSD scalar-A-per-head form:
        h: (batch, nheads, headdim, d_state) mlx.core.array

    Frozen dataclass per canonical immutability; the wrapped mlx.core.array is
    immutable by construction (MLX functional API).
    """

    h: Any  # mlx.core.array — typed as Any to avoid MLX import at module-top

    def __post_init__(self) -> None:
        # Defer MLX type check to runtime (no module-top import).
        if MLX_AVAILABLE:
            import mlx.core as mx  # noqa: PLC0415
            if not isinstance(self.h, mx.array):
                raise TypeError(
                    f"Mamba2SSDMLXState.h must be mlx.core.array, got {type(self.h).__name__}"
                )
            if self.h.ndim != 4:
                raise ValueError(
                    f"Mamba2SSDMLXState.h must have shape (B, nheads, headdim, d_state); "
                    f"got ndim={self.h.ndim} shape={self.h.shape}"
                )
        # If MLX is unavailable, the state can still be constructed for testing
        # via numpy arrays (the dataclass is duck-typed).


def mamba2_ssd_init_state_mlx(
    *,
    batch_size: int,
    nheads: int,
    headdim: int,
    d_state: int,
    dtype: Any | None = None,
) -> Mamba2SSDMLXState:
    """Allocate zero initial state per Mamba-2 SSD canonical (Dao+Gu 2024 §4).

    Args:
        batch_size, nheads, headdim, d_state: as numpy sister.
        dtype: mlx.core.Dtype; defaults to mx.float32.

    Returns:
        :class:`Mamba2SSDMLXState` with h = zeros((batch, nheads, headdim, d_state)).

    Raises:
        ImportError: if MLX not available on this host.
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend requested but mlx not importable. Install via "
            "`uv pip install mlx` (Darwin ARM64 / Apple Silicon only)."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if nheads <= 0:
        raise ValueError(f"nheads must be > 0, got {nheads}")
    if headdim <= 0:
        raise ValueError(f"headdim must be > 0, got {headdim}")
    if d_state <= 0:
        raise ValueError(f"d_state must be > 0, got {d_state}")
    import mlx.core as mx  # noqa: PLC0415
    if dtype is None:
        dtype = mx.float32
    h = mx.zeros((batch_size, nheads, headdim, d_state), dtype=dtype)
    return Mamba2SSDMLXState(h=h)


def mamba2_ssd_step_mlx(
    *,
    state: Mamba2SSDMLXState,
    x_t: Any,
    A_log: Any,
    B_t: Any,
    C_t: Any,
    dt_t: Any,
    D: Any | None = None,
) -> Tuple[Mamba2SSDMLXState, Any]:
    """Single-step canonical Mamba-2 SSD recurrence (MLX impl byte-stable vs numpy).

    Math contract matches :func:`.numpy_backend.mamba2_ssd_step_numpy` exactly
    within MLX float32 numerical tolerance (max_abs < 3e-5 per drift discipline).

    Args mirror numpy sister; tensors are mlx.core.array.
    Returns mirror numpy sister; tensors are mlx.core.array.
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend invoked but mlx not importable. Install via "
            "`uv pip install mlx` (Darwin ARM64 / Apple Silicon only)."
        )
    import mlx.core as mx  # noqa: PLC0415

    if not isinstance(state, Mamba2SSDMLXState):
        raise TypeError(
            f"state must be Mamba2SSDMLXState, got {type(state).__name__}"
        )
    h = state.h
    B, nheads, headdim, d_state = h.shape

    # Discretize: A = -exp(A_log); α_t = exp(dt_t * A)
    A_neg = -mx.exp(A_log)  # (nheads,)
    # Broadcast (nheads,) → (1, nheads) for (B, nheads) * (1, nheads) = (B, nheads)
    alpha_t = mx.exp(dt_t * A_neg[None, :])  # (B, nheads)
    # Broadcast to (B, nheads, 1, 1) for state update
    alpha_t_bcast = alpha_t[:, :, None, None]  # (B, nheads, 1, 1)

    # B_bar_t = dt_t * B_t (ZOH approx)
    dt_t_bcast = dt_t[:, :, None]  # (B, nheads, 1)
    B_bar_t = dt_t_bcast * B_t  # (B, nheads, d_state)

    # State update: h_t = α_t · h_{t-1} + B_bar_t ⊗ x_t
    Bx = x_t[:, :, :, None] * B_bar_t[:, :, None, :]  # (B, nheads, headdim, d_state)
    h_next = alpha_t_bcast * h + Bx  # (B, nheads, headdim, d_state)

    # Output: y_t = sum_{d_state} (h_t * C_t)
    # mlx.core.einsum exists per the canonical primitive set (mlx ≥ 0.18+).
    # Fallback impl for older MLX: explicit broadcast + sum.
    try:
        y_t = mx.einsum("bhdk,bhk->bhd", h_next, C_t)  # (B, nheads, headdim)
    except (AttributeError, NotImplementedError):
        # Older MLX versions: explicit broadcast (B, nheads, headdim, d_state) *
        # (B, nheads, 1, d_state) → sum over d_state.
        y_t = (h_next * C_t[:, :, None, :]).sum(axis=-1)  # (B, nheads, headdim)

    # Optional skip connection: y_t += D · x_t
    if D is not None:
        y_t = y_t + D[None, :, :] * x_t  # (B, nheads, headdim)

    return Mamba2SSDMLXState(h=h_next), y_t


def mamba2_ssd_forward_sequence_mlx(
    *,
    x_seq: Any,
    A_log: Any,
    B_seq: Any,
    C_seq: Any,
    dt_seq: Any,
    D: Any | None = None,
    initial_state: Mamba2SSDMLXState | None = None,
) -> Tuple[Mamba2SSDMLXState, Any]:
    """Forward over a length-L sequence; sequential SSD scan (MLX).

    Math contract matches :func:`.numpy_backend.mamba2_ssd_forward_sequence_numpy`
    exactly within MLX float32 numerical tolerance.

    Args / returns mirror numpy sister; tensors are mlx.core.array.
    """
    if not MLX_AVAILABLE:
        raise ImportError(
            "MLX backend invoked but mlx not importable. Install via "
            "`uv pip install mlx` (Darwin ARM64 / Apple Silicon only)."
        )
    import mlx.core as mx  # noqa: PLC0415

    if x_seq.ndim != 4:
        raise ValueError(
            f"x_seq must be 4D (B, L, nheads, headdim); got shape {x_seq.shape}"
        )
    B, L, nheads, headdim = x_seq.shape
    d_state = B_seq.shape[3]

    if initial_state is None:
        state = mamba2_ssd_init_state_mlx(
            batch_size=B, nheads=nheads, headdim=headdim, d_state=d_state,
            dtype=x_seq.dtype,
        )
    else:
        state = initial_state

    # Sequential scan — MLX backend mirrors numpy reference math.
    # A Metal-kernel chunked-scan optimization is a future sister wave per
    # CLAUDE.md "Forbidden premature KILL without research exhaustion" +
    # the optimization standing directive 2026-05-29; canonical scope here
    # is mathematical truth + byte-stable parity vs numpy reference.
    y_out_list: list[Any] = []
    for t in range(L):
        state, y_t = mamba2_ssd_step_mlx(
            state=state,
            x_t=x_seq[:, t, :, :],
            A_log=A_log,
            B_t=B_seq[:, t, :, :],
            C_t=C_seq[:, t, :, :],
            dt_t=dt_seq[:, t, :],
            D=D,
        )
        y_out_list.append(y_t)

    # mx.stack along axis=1 for sequence dim
    y_out = mx.stack(y_out_list, axis=1)  # (B, L, nheads, headdim)
    # Force evaluation per MLX lazy semantics (canonical for byte-stable comparison).
    mx.eval(y_out, state.h)
    return state, y_out
