# SPDX-License-Identifier: MIT
"""Canonical PyTorch impl of Mamba-2 SSD selective state-space recurrence.

This backend mirrors :mod:`tac.substrates._shared.mamba2_ssd.numpy_backend`
math byte-for-byte (within float32 numerical tolerance; max_abs < 3e-5 per
the canonical drift discipline) using PyTorch tensors so gradients flow
through the recurrence for substrate-aware training paths.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable:
PyTorch is the canonical contest-resolution framework; this backend is the
canonical training surface for paid Modal / Vast.ai / Lightning dispatches.

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)"
+ Catalog #1 + Catalog #205 sister: this backend does NOT silently auto-fall-
back to MPS; the caller chooses device explicitly. If torch.cuda.is_available()
is False AND caller passes device='cuda', PyTorch raises explicitly (not this
helper's responsibility — torch.Tensor.to('cuda') raises canonically).

Math fidelity vs numpy reference
---------------------------------

Sequential step / forward routines mirror numpy reference math exactly:

    α_t = exp(dt_t * (-exp(A_log)))           # (B, nheads); broadcast across (headdim, d_state)
    h_t = α_t · h_{t-1} + dt_t · (B_t ⊗ x_t)  # state update
    y_t = einsum("bhdk,bhk->bhd", h_t, C_t)   # output projection over d_state
    y_t += D · x_t                              # optional skip connection

Documented adaptations per CLAUDE.md "Forbidden empirical-claim-without-
evidence-tag" + 5-axis taxonomy:

  1. **Axis 4 (math)**: einsum is mathematically equivalent to the numpy
     sum-with-broadcast pattern; we use einsum for PyTorch idiomaticity.
  2. **Axis 4 (math)**: backend produces float32 by default; PyTorch float64
     precision available via dtype kwarg for high-precision testing.
  3. **Axis 3 (problem space)**: gradient-preserving (no detach in forward).
  4. **Axis 5 (data)**: device + dtype routing per caller (CPU / CUDA / MPS).
     MPS results are non-promotable per CLAUDE.md "MPS auth eval is NOISE"
     non-negotiable + Catalog #1; this is a runtime concern not a math concern.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L8
(eval-roundtrip-aware): this backend is the canonical training-time surface
that downstream eval-roundtrip + Hinton-KL distillation pipelines consume.

Cross-references:
  * Sister numpy reference: :mod:`.numpy_backend`
  * Sister MLX impl: :mod:`.mlx_backend`
  * Tri-backend dispatch: :mod:`.` (package __init__)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "Mamba2SSDPyTorchState",
    "mamba2_ssd_init_state_pytorch",
    "mamba2_ssd_step_pytorch",
    "mamba2_ssd_forward_sequence_pytorch",
]


@dataclass(frozen=True)
class Mamba2SSDPyTorchState:
    """Externalized SSD recurrent state for the PyTorch backend.

    Shape contract per Dao+Gu 2024 §4 SSD scalar-A-per-head form:
        h: (batch, nheads, headdim, d_state) torch.Tensor

    Frozen dataclass per canonical immutability per Catalog #131/#138 sister
    discipline (note: the underlying torch.Tensor is mutable in-place but the
    dataclass wrapper is frozen so callers cannot rebind the slot).
    """

    h: torch.Tensor  # (batch, nheads, headdim, d_state)

    def __post_init__(self) -> None:
        if not isinstance(self.h, torch.Tensor):
            raise TypeError(
                f"Mamba2SSDPyTorchState.h must be torch.Tensor, got {type(self.h).__name__}"
            )
        if self.h.dim() != 4:
            raise ValueError(
                f"Mamba2SSDPyTorchState.h must have shape (B, nheads, headdim, d_state); "
                f"got ndim={self.h.dim()} shape={tuple(self.h.shape)}"
            )


def mamba2_ssd_init_state_pytorch(
    *,
    batch_size: int,
    nheads: int,
    headdim: int,
    d_state: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Mamba2SSDPyTorchState:
    """Allocate zero initial state per Mamba-2 SSD canonical (Dao+Gu 2024 §4).

    Args:
        batch_size, nheads, headdim, d_state: as numpy sister.
        device: torch.device or string; default 'cpu'.
        dtype: torch dtype; default float32.

    Returns:
        :class:`Mamba2SSDPyTorchState` with h = zeros((batch, nheads, headdim, d_state)).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if nheads <= 0:
        raise ValueError(f"nheads must be > 0, got {nheads}")
    if headdim <= 0:
        raise ValueError(f"headdim must be > 0, got {headdim}")
    if d_state <= 0:
        raise ValueError(f"d_state must be > 0, got {d_state}")
    h = torch.zeros((batch_size, nheads, headdim, d_state), device=device, dtype=dtype)
    return Mamba2SSDPyTorchState(h=h)


def mamba2_ssd_step_pytorch(
    *,
    state: Mamba2SSDPyTorchState,
    x_t: torch.Tensor,
    A_log: torch.Tensor,
    B_t: torch.Tensor,
    C_t: torch.Tensor,
    dt_t: torch.Tensor,
    D: torch.Tensor | None = None,
) -> Tuple[Mamba2SSDPyTorchState, torch.Tensor]:
    """Single-step canonical Mamba-2 SSD recurrence (PyTorch impl byte-stable vs numpy).

    Math contract matches :func:`.numpy_backend.mamba2_ssd_step_numpy` exactly.
    See that function's docstring for the full mathematical specification.

    Args:
        state: prior :class:`Mamba2SSDPyTorchState`.
        x_t: per-step input (B, nheads, headdim).
        A_log: SSD A_log (nheads,).
        B_t: per-step B (B, nheads, d_state).
        C_t: per-step C (B, nheads, d_state).
        dt_t: per-step step-size post-softplus (B, nheads).
        D: optional skip (nheads, headdim).

    Returns:
        Tuple (next_state, y_t) with shapes matching numpy sister.
    """
    if not isinstance(state, Mamba2SSDPyTorchState):
        raise TypeError(
            f"state must be Mamba2SSDPyTorchState, got {type(state).__name__}"
        )
    h = state.h
    B, nheads, headdim, d_state = h.shape
    if not (isinstance(x_t, torch.Tensor) and tuple(x_t.shape) == (B, nheads, headdim)):
        raise ValueError(
            f"x_t shape {tuple(x_t.shape) if isinstance(x_t, torch.Tensor) else None} "
            f"expected ({B}, {nheads}, {headdim})"
        )
    if not (isinstance(A_log, torch.Tensor) and tuple(A_log.shape) == (nheads,)):
        raise ValueError(
            f"A_log shape {tuple(A_log.shape) if isinstance(A_log, torch.Tensor) else None} "
            f"expected ({nheads},)"
        )
    if not (isinstance(B_t, torch.Tensor) and tuple(B_t.shape) == (B, nheads, d_state)):
        raise ValueError(
            f"B_t shape {tuple(B_t.shape) if isinstance(B_t, torch.Tensor) else None} "
            f"expected ({B}, {nheads}, {d_state})"
        )
    if not (isinstance(C_t, torch.Tensor) and tuple(C_t.shape) == (B, nheads, d_state)):
        raise ValueError(
            f"C_t shape {tuple(C_t.shape) if isinstance(C_t, torch.Tensor) else None} "
            f"expected ({B}, {nheads}, {d_state})"
        )
    if not (isinstance(dt_t, torch.Tensor) and tuple(dt_t.shape) == (B, nheads)):
        raise ValueError(
            f"dt_t shape {tuple(dt_t.shape) if isinstance(dt_t, torch.Tensor) else None} "
            f"expected ({B}, {nheads})"
        )

    # Discretize: A = -exp(A_log); α_t = exp(dt_t * A)
    A_neg = -torch.exp(A_log)  # (nheads,)
    alpha_t = torch.exp(dt_t * A_neg.unsqueeze(0))  # (B, nheads)
    alpha_t_bcast = alpha_t.unsqueeze(-1).unsqueeze(-1)  # (B, nheads, 1, 1)

    # B_bar_t = dt_t * B_t (ZOH approx)
    dt_t_bcast = dt_t.unsqueeze(-1)  # (B, nheads, 1)
    B_bar_t = dt_t_bcast * B_t  # (B, nheads, d_state)

    # State update: h_t = α_t · h_{t-1} + B_bar_t ⊗ x_t
    Bx = x_t.unsqueeze(-1) * B_bar_t.unsqueeze(2)  # (B, nheads, headdim, d_state)
    h_next = alpha_t_bcast * h + Bx  # (B, nheads, headdim, d_state)

    # Output: y_t = sum_{d_state} (h_t * C_t)
    y_t = torch.einsum("bhdk,bhk->bhd", h_next, C_t)  # (B, nheads, headdim)

    # Optional skip connection
    if D is not None:
        if not (isinstance(D, torch.Tensor) and tuple(D.shape) == (nheads, headdim)):
            raise ValueError(
                f"D shape {tuple(D.shape) if isinstance(D, torch.Tensor) else None} "
                f"expected ({nheads}, {headdim})"
            )
        y_t = y_t + D.unsqueeze(0) * x_t  # (B, nheads, headdim)

    return Mamba2SSDPyTorchState(h=h_next), y_t


def mamba2_ssd_forward_sequence_pytorch(
    *,
    x_seq: torch.Tensor,
    A_log: torch.Tensor,
    B_seq: torch.Tensor,
    C_seq: torch.Tensor,
    dt_seq: torch.Tensor,
    D: torch.Tensor | None = None,
    initial_state: Mamba2SSDPyTorchState | None = None,
) -> Tuple[Mamba2SSDPyTorchState, torch.Tensor]:
    """Forward over a length-L sequence; sequential SSD scan (PyTorch).

    Math contract matches :func:`.numpy_backend.mamba2_ssd_forward_sequence_numpy`
    exactly. Gradient flow: all forward operations are torch primitives; gradients
    flow through ``h_next`` recurrence + output projection without detach.

    Args:
        x_seq: input sequence (B, L, nheads, headdim).
        A_log: (nheads,).
        B_seq: per-step B (B, L, nheads, d_state).
        C_seq: per-step C (B, L, nheads, d_state).
        dt_seq: per-step dt (B, L, nheads).
        D: optional skip (nheads, headdim).
        initial_state: optional :class:`Mamba2SSDPyTorchState`; defaults to zero.

    Returns:
        Tuple (final_state, y_seq) where y_seq has shape (B, L, nheads, headdim).
    """
    if not isinstance(x_seq, torch.Tensor) or x_seq.dim() != 4:
        raise ValueError(
            f"x_seq must be 4D torch.Tensor (B, L, nheads, headdim); got "
            f"{tuple(x_seq.shape) if isinstance(x_seq, torch.Tensor) else None}"
        )
    B, L, nheads, headdim = x_seq.shape
    d_state = B_seq.shape[3]

    if initial_state is None:
        state = mamba2_ssd_init_state_pytorch(
            batch_size=B, nheads=nheads, headdim=headdim, d_state=d_state,
            device=x_seq.device, dtype=x_seq.dtype,
        )
    else:
        state = initial_state

    # Sequential scan — PyTorch backend mirrors numpy reference math exactly.
    # A chunked-scan throughput optimization (Triton kernel per state-spaces/mamba)
    # is a future optimization sister wave; this canonical impl is mathematical truth.
    y_out_list: list[torch.Tensor] = []
    for t in range(L):
        state, y_t = mamba2_ssd_step_pytorch(
            state=state,
            x_t=x_seq[:, t, :, :],
            A_log=A_log,
            B_t=B_seq[:, t, :, :],
            C_t=C_seq[:, t, :, :],
            dt_t=dt_seq[:, t, :],
            D=D,
        )
        y_out_list.append(y_t)

    y_out = torch.stack(y_out_list, dim=1)  # (B, L, nheads, headdim)
    return state, y_out
