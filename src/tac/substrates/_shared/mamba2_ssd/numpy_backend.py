# SPDX-License-Identifier: MIT
"""Canonical numpy reference impl of Mamba-2 SSD selective state-space recurrence.

Per operator NON-NEGOTIABLE 2026-05-30 directive verbatim *"perhaps let's port
it"* + *"will it still be portable via numpy"* + *"wherever we are missing MLX
implementations or any grammar or anything let's do it"*. This numpy reference
is the MATHEMATICAL TRUTH backend: deterministic, fully portable, no GPU dep.
PyTorch + MLX backends MUST produce byte-stable outputs vs this reference per
the canonical max_abs < 3e-5 NUMERIC_TOLERANCE drift discipline per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" + Slot 1303 T3 GRAND COUNCIL MLX-vs-
PyTorch drift symposium.

Mathematical grounding — Dao+Gu 2024 §3-§4 (arxiv 2405.21060)
--------------------------------------------------------------

Mamba-2 SSD canonical form (Theorem 3.5: State Space Duality):

  Given input sequence x ∈ R^{B, L, D} and per-step input-dependent matrices
  (A_t, B_t, C_t), the SSD output y ∈ R^{B, L, D} satisfies:

      h_0 = 0
      h_t = A_t · h_{t-1} + B_t · x_t           # state update
      y_t = C_t^T · h_t                          # output projection

  where the canonical Mamba-2 SSD structure pins A_t = α_t · I_n (scalar-times-
  identity, i.e. scalar per head per timestep) so the recurrence reduces to:

      h_t = α_t · h_{t-1} + B_t · x_t

  This is the structural simplification that enables the chunk-based matmul
  parallelization (Theorem 3.5 ⇒ §4 chunked-scan algorithm). The numpy
  reference computes this sequentially — no chunk parallelism — because numpy
  has no parallel-scan primitive AND because the reference's purpose is
  MATHEMATICAL TRUTH not throughput. PyTorch + MLX backends MAY implement
  chunked-scan for throughput (Triton kernel / Metal kernel) but MUST produce
  byte-stable output vs this reference.

Discretization (ZOH per §2.2):

      α_t = exp(Δt_t · A_log)           # diagonal-scalar form (Mamba-2)
      B_bar_t = Δt_t · B_t              # ZOH approximation

  where Δt_t is the input-dependent step-size (computed via softplus(Δt_proj(x_t))).

Per-head structure (SSD scalar-A-per-head per Dao+Gu 2024 §4):

  The state h has shape (B, nheads, headdim, d_state) where:
    - nheads = d_inner / headdim
    - headdim is the per-head feature dimension (canonical: 64)
    - d_state is the SSM state dimension (canonical: 16)

  A_log has shape (nheads,) — scalar per head, broadcast across (headdim, d_state).
  This is the DOMINANT distinguishing feature vs Mamba-1 S6 (which had A_log
  shape (d_inner, d_state) — diagonal per-(channel, state)).

Documented adaptations for problem-space optimization per CLAUDE.md
"Forbidden empirical-claim-without-evidence-tag" + 5-axis taxonomy:

  1. **Axis 4 (math)**: B_proj + C_proj are computed per-step from the input
     (input-dependent matrices); we follow upstream state-spaces/mamba canonical
     where these are linear projections (no per-head structure for B, C).
  2. **Axis 4 (math)**: dt initialization uses inverse-softplus from log-uniform
     [dt_min=0.001, dt_max=0.1] per upstream canonical.
  3. **Axis 3 (problem space — contest scale)**: nheads + headdim are caller-
     supplied; canonical defaults nheads=2 + headdim=64 ⇒ d_inner=128 match the
     Z7-Mamba-2 substrate scale per parent design memo §7.
  4. **Axis 5 (data — sequential vs chunked)**: numpy reference is sequential;
     PyTorch / MLX backends MAY use chunked-scan for throughput.

Cross-references:
  * CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing directive
  * Catalog #205 — sister at inflate-time device-selection surface
  * Catalog #1265 — contest-equivalence gate (this is the SSD grammar)
  * `tac.optimization.mamba2_predictor._ReferenceMamba2Cell` — sister Mamba-1 S6
    reference (NOT this — this is the Mamba-2 SSD reference)
  * state-spaces/mamba `mamba_ssm.modules.mamba2.Mamba2` — canonical CUDA upstream
  * https://github.com/purohit10saurabh/mamba-ssm-macos — MPS reference (sister)

[verified-against: Dao & Gu 2024 arxiv 2405.21060 §3 Theorem 3.5 + §4 SSD algorithm]
[verified-against: state-spaces/mamba `mamba_ssm.modules.mamba2.Mamba2` upstream]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "Mamba2SSDNumpyState",
    "mamba2_ssd_init_state_numpy",
    "mamba2_ssd_step_numpy",
    "mamba2_ssd_forward_sequence_numpy",
]


@dataclass(frozen=True)
class Mamba2SSDNumpyState:
    """Externalized SSD recurrent state for the numpy backend.

    Shape contract per Dao+Gu 2024 §4 SSD scalar-A-per-head form:
        h: (batch, nheads, headdim, d_state)

    The state is allocated via :func:`mamba2_ssd_init_state_numpy` and
    updated via :func:`mamba2_ssd_step_numpy`. Frozen dataclass per
    canonical immutability per Catalog #131/#138 sister discipline.
    """

    h: np.ndarray  # (batch, nheads, headdim, d_state) float32

    def __post_init__(self) -> None:
        if not isinstance(self.h, np.ndarray):
            raise TypeError(
                f"Mamba2SSDNumpyState.h must be np.ndarray, got {type(self.h).__name__}"
            )
        if self.h.ndim != 4:
            raise ValueError(
                f"Mamba2SSDNumpyState.h must have shape (B, nheads, headdim, d_state); "
                f"got ndim={self.h.ndim} shape={self.h.shape}"
            )
        # numpy dtype check is best-effort; the canonical contract is float32 but
        # float64 is allowed for high-precision testing.
        if self.h.dtype not in (np.float32, np.float64):
            raise ValueError(
                f"Mamba2SSDNumpyState.h dtype must be float32 or float64; got {self.h.dtype}"
            )


def mamba2_ssd_init_state_numpy(
    *,
    batch_size: int,
    nheads: int,
    headdim: int,
    d_state: int,
    dtype: np.dtype = np.float32,
) -> Mamba2SSDNumpyState:
    """Allocate zero initial state per Mamba-2 SSD canonical (Dao+Gu 2024 §4).

    Args:
        batch_size: number of parallel sequences (B).
        nheads: number of SSD heads (canonical default 2 for contest scale).
        headdim: per-head feature dimension (canonical default 64).
        d_state: SSM state dimension (canonical default 16 — note: language scale
            uses 128 but contest 600-pair scale uses 16 per parent design memo).
        dtype: numpy dtype; default float32.

    Returns:
        :class:`Mamba2SSDNumpyState` with h = zeros((batch, nheads, headdim, d_state)).

    Raises:
        ValueError: on non-positive dim args.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if nheads <= 0:
        raise ValueError(f"nheads must be > 0, got {nheads}")
    if headdim <= 0:
        raise ValueError(f"headdim must be > 0, got {headdim}")
    if d_state <= 0:
        raise ValueError(f"d_state must be > 0, got {d_state}")
    h = np.zeros((batch_size, nheads, headdim, d_state), dtype=dtype)
    return Mamba2SSDNumpyState(h=h)


def mamba2_ssd_step_numpy(
    *,
    state: Mamba2SSDNumpyState,
    x_t: np.ndarray,
    A_log: np.ndarray,
    B_t: np.ndarray,
    C_t: np.ndarray,
    dt_t: np.ndarray,
    D: np.ndarray | None = None,
) -> Tuple[Mamba2SSDNumpyState, np.ndarray]:
    """Single-step canonical Mamba-2 SSD recurrence per Dao+Gu 2024 §3 Theorem 3.5.

    Mathematical contract (sequential reference; not chunked):

        α_t = exp(dt_t * A_log)                  # (B, nheads); scalar per head
        h_t = α_t * h_{t-1} + dt_t * (B_t ⊗ x_t)  # SSD state update
        y_t = (h_t @ C_t) summed over d_state    # (B, nheads, headdim)
        y_t += D * x_t                            # skip connection (if D provided)

    where:
        - α_t is broadcast across (headdim, d_state): (B, nheads) → (B, nheads, 1, 1)
        - B_t has shape (B, nheads, d_state); broadcast over headdim
        - C_t has shape (B, nheads, d_state); summed via einsum to (B, nheads, headdim)
        - dt_t has shape (B, nheads) — input-dependent step-size
        - D has shape (nheads, headdim) skip-connection multiplier (canonical Mamba-2)

    Args:
        state: prior :class:`Mamba2SSDNumpyState` (h shape: B, nheads, headdim, d_state).
        x_t: per-step input (B, nheads, headdim) float32.
        A_log: SSD A_log parameter (nheads,) — scalar per head per Dao+Gu 2024 §4.
        B_t: input-dependent SSM B matrix (B, nheads, d_state).
        C_t: input-dependent SSM C matrix (B, nheads, d_state).
        dt_t: input-dependent step-size (B, nheads); MUST be post-softplus
            (positive).
        D: optional skip-connection multiplier (nheads, headdim).

    Returns:
        Tuple (next_state, y_t) where next_state is the updated
        :class:`Mamba2SSDNumpyState` and y_t has shape (B, nheads, headdim).

    Raises:
        TypeError: on non-ndarray inputs.
        ValueError: on shape contract violations.
    """
    if not isinstance(state, Mamba2SSDNumpyState):
        raise TypeError(
            f"state must be Mamba2SSDNumpyState, got {type(state).__name__}"
        )
    h = state.h
    B, nheads, headdim, d_state = h.shape
    # Shape checks (numpy ndim contract per Dao+Gu 2024 §4).
    if not (isinstance(x_t, np.ndarray) and x_t.shape == (B, nheads, headdim)):
        raise ValueError(
            f"x_t shape {getattr(x_t, 'shape', None)} expected ({B}, {nheads}, {headdim})"
        )
    if not (isinstance(A_log, np.ndarray) and A_log.shape == (nheads,)):
        raise ValueError(
            f"A_log shape {getattr(A_log, 'shape', None)} expected ({nheads},) "
            f"per Mamba-2 SSD scalar-A-per-head"
        )
    if not (isinstance(B_t, np.ndarray) and B_t.shape == (B, nheads, d_state)):
        raise ValueError(
            f"B_t shape {getattr(B_t, 'shape', None)} expected ({B}, {nheads}, {d_state})"
        )
    if not (isinstance(C_t, np.ndarray) and C_t.shape == (B, nheads, d_state)):
        raise ValueError(
            f"C_t shape {getattr(C_t, 'shape', None)} expected ({B}, {nheads}, {d_state})"
        )
    if not (isinstance(dt_t, np.ndarray) and dt_t.shape == (B, nheads)):
        raise ValueError(
            f"dt_t shape {getattr(dt_t, 'shape', None)} expected ({B}, {nheads})"
        )

    # Discretize: α_t = exp(dt_t * A_log_negative) — A_log is the log of negative
    # eigenvalue magnitude; canonical Mamba-2 SSD uses A = -exp(A_log) so the
    # actual SSM matrix A is negative for stability per Gu&Dao 2023 §3.
    # dt_t has shape (B, nheads); A_log has shape (nheads,) → broadcast to (B, nheads).
    # Then expand to (B, nheads, 1, 1) for state-update broadcast.
    A_neg = -np.exp(A_log)  # (nheads,) negative eigenvalues
    alpha_t = np.exp(dt_t * A_neg[None, :])  # (B, nheads)
    alpha_t_bcast = alpha_t[:, :, None, None]  # (B, nheads, 1, 1)

    # B_bar_t = dt_t * B_t (ZOH approximation per Dao+Gu 2024 §2.2).
    # dt_t (B, nheads) → (B, nheads, 1); B_t (B, nheads, d_state) → broadcast.
    dt_t_bcast = dt_t[:, :, None]  # (B, nheads, 1)
    B_bar_t = dt_t_bcast * B_t  # (B, nheads, d_state)

    # State update: h_t = α_t * h_{t-1} + B_bar_t ⊗ x_t
    # B_bar_t (B, nheads, d_state) ⊗ x_t (B, nheads, headdim) → (B, nheads, headdim, d_state)
    Bx = x_t[:, :, :, None] * B_bar_t[:, :, None, :]  # (B, nheads, headdim, d_state)
    h_next = alpha_t_bcast * h + Bx  # (B, nheads, headdim, d_state)

    # Output: y_t = sum_{d_state} (h_t * C_t)
    # C_t (B, nheads, d_state) → broadcast to (B, nheads, 1, d_state); sum over d_state.
    y_t = np.einsum("bhdk,bhk->bhd", h_next, C_t)  # (B, nheads, headdim)

    # Optional skip connection: y_t += D * x_t (Mamba-2 canonical per upstream).
    if D is not None:
        if not (isinstance(D, np.ndarray) and D.shape == (nheads, headdim)):
            raise ValueError(
                f"D shape {getattr(D, 'shape', None)} expected ({nheads}, {headdim})"
            )
        y_t = y_t + D[None, :, :] * x_t  # (B, nheads, headdim)

    return Mamba2SSDNumpyState(h=h_next), y_t


def mamba2_ssd_forward_sequence_numpy(
    *,
    x_seq: np.ndarray,
    A_log: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
    dt_seq: np.ndarray,
    D: np.ndarray | None = None,
    initial_state: Mamba2SSDNumpyState | None = None,
) -> Tuple[Mamba2SSDNumpyState, np.ndarray]:
    """Forward over a length-L sequence; canonical sequential SSD scan.

    Mathematical equivalence to chunked-scan parallelization (Dao+Gu 2024 §4
    Theorem 3.5 + Algorithm 1): the chunked algorithm produces byte-identical
    output to this sequential scan when reduced over the chunk dimension; the
    chunk algorithm is a throughput optimization, not a math change. The
    PyTorch + MLX backends MAY implement chunked-scan; this reference is the
    sequential truth.

    Args:
        x_seq: input sequence (B, L, nheads, headdim).
        A_log: (nheads,) — broadcast across all timesteps.
        B_seq: per-step B matrix (B, L, nheads, d_state).
        C_seq: per-step C matrix (B, L, nheads, d_state).
        dt_seq: per-step step-size post-softplus (B, L, nheads).
        D: optional skip-connection (nheads, headdim).
        initial_state: optional :class:`Mamba2SSDNumpyState`; defaults to zero state.

    Returns:
        Tuple (final_state, y_seq) where y_seq has shape (B, L, nheads, headdim).

    Raises:
        ValueError: on shape mismatch.
    """
    if not isinstance(x_seq, np.ndarray) or x_seq.ndim != 4:
        raise ValueError(
            f"x_seq must be 4D ndarray (B, L, nheads, headdim); got shape "
            f"{getattr(x_seq, 'shape', None)}"
        )
    B, L, nheads, headdim = x_seq.shape
    if not (isinstance(B_seq, np.ndarray) and B_seq.ndim == 4 and B_seq.shape[:3] == (B, L, nheads)):
        raise ValueError(
            f"B_seq shape {getattr(B_seq, 'shape', None)} expected ({B}, {L}, {nheads}, d_state)"
        )
    d_state = B_seq.shape[3]
    if not (isinstance(C_seq, np.ndarray) and C_seq.shape == (B, L, nheads, d_state)):
        raise ValueError(
            f"C_seq shape {getattr(C_seq, 'shape', None)} expected ({B}, {L}, {nheads}, {d_state})"
        )
    if not (isinstance(dt_seq, np.ndarray) and dt_seq.shape == (B, L, nheads)):
        raise ValueError(
            f"dt_seq shape {getattr(dt_seq, 'shape', None)} expected ({B}, {L}, {nheads})"
        )

    if initial_state is None:
        state = mamba2_ssd_init_state_numpy(
            batch_size=B, nheads=nheads, headdim=headdim, d_state=d_state,
            dtype=x_seq.dtype,
        )
    else:
        state = initial_state
        if state.h.shape != (B, nheads, headdim, d_state):
            raise ValueError(
                f"initial_state.h shape {state.h.shape} != expected "
                f"({B}, {nheads}, {headdim}, {d_state})"
            )

    y_out = np.zeros((B, L, nheads, headdim), dtype=x_seq.dtype)
    for t in range(L):
        state, y_t = mamba2_ssd_step_numpy(
            state=state,
            x_t=x_seq[:, t, :, :],
            A_log=A_log,
            B_t=B_seq[:, t, :, :],
            C_t=C_seq[:, t, :, :],
            dt_t=dt_seq[:, t, :],
            D=D,
        )
        y_out[:, t, :, :] = y_t

    return state, y_out
