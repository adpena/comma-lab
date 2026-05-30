# SPDX-License-Identifier: MIT
"""Canonical Mamba-2 SSD tri-backend helper (MLX / PyTorch / numpy).

Per operator NON-NEGOTIABLE 2026-05-30 directive chain:
  * *"perhaps let's port it"* — port Mamba-2 SSD to MLX
  * *"it may be worth it in the long run"* — multi-week canonical lift
  * *"will it still be portable via numpy"* — YES, this helper has 3 backends
  * *"research online to see if any implementations exist in any OSS anywhere"*
    — verified state-spaces/mamba + purohit10saurabh/mamba-ssm-macos + alxndrTL/mamba.py
  * *"wherever we are missing MLX implementations or any grammar or anything let's do it"*

This helper is the canonical Mamba-2 SSD (Dao+Gu 2024 §3-§4 — State-Space
Duality) recurrent SSM primitive with three byte-stable backends:

  * **NUMPY** (mathematical truth; portable; deterministic)
  * **PYTORCH** (contest-resolution training surface; gradient-preserving)
  * **MLX** (M5 Max training surface per CLAUDE.md 8th standing directive)

The three backends produce byte-stable output within float32 numerical
tolerance (max_abs < 3e-5 per Slot 1303 T3 GRAND COUNCIL MLX-vs-PyTorch drift
discipline + Slot 1255 PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING).

Tri-backend dispatch
--------------------

Per :mod:`tac.framework_agnostic.decorators` canonical pattern, this helper
exposes :func:`compute_mamba2_ssd_forward` which routes to the correct backend
via the canonical :class:`Backend` selection cascade per Catalog #205 sister:

    1. Explicit ``backend=`` kwarg (highest precedence)
    2. ``PACT_FRAMEWORK_BACKEND`` env var
    3. Platform priority (Darwin ARM64 → MLX; Linux → PYTORCH; else NUMPY)

Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)" +
Catalog #1 sister: callers MUST handle :class:`BackendUnavailableError` per
the canonical cascade discipline; this helper does NOT silently auto-fall-back
to a non-promotable backend.

Distinguishing features per CLAUDE.md "HNeRV / leaderboard-implementation parity
discipline" L18 (PixelShuffle + bilinear-skip + sin) sister — at the SSM layer:

  * **Mamba-2 SSD scalar-A-per-head** (per Dao+Gu 2024 §4) NOT Mamba-1 S6
    diagonal-A-per-channel — the canonical chunked-scan parallelization target.
  * **Externalized state pass-through** — caller manages h state for checkpointable
    deterministic recurrence (sister of :func:`Mamba2Predictor.step_externalized_state`
    + Z8 ``DeterministicStateUpdate`` Protocol).
  * **6-hook wire-in declaration** per Catalog #125:
    - hook #1 sensitivity-map: ACTIVE (per-tensor gradient norms surface)
    - hook #2 Pareto constraint: ACTIVE (SSM state-entropy constraint)
    - hook #3 bit-allocator: ACTIVE (per-tensor A_log + B_proj + C_proj weights)
    - hook #4 cathedral autopilot dispatch: ACTIVE (auto-discovered via sister
      consumer per Catalog #335)
    - hook #5 continual-learning posterior: ACTIVE (anchor accumulation)
    - hook #6 probe-disambiguator: ACTIVE (3-backend parity IS the disambiguator)

Apparatus mutation chain per CLAUDE.md "Results must become system intelligence":

  * Canonical equation: ``mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1``
    (registered via :mod:`tac.canonical_equations` per Catalog #344)
  * Lane registry: ``lane_mamba2_ssd_mlx_port_tri_backend_20260530`` L1
  * Catalog #348 retroactive sweep: documents the orphan-mamba2-mlx-impl bug
    class that this helper extincts structurally

Cross-references (canonical helpers consumed):
  * :mod:`tac.framework_agnostic.backend` — Backend selection
  * :mod:`tac.framework_agnostic.decorators` — decorator dispatch
  * :mod:`tac.optimization.mamba2_predictor` — Mamba-1 S6 sister (existing canonical;
    this helper is the Mamba-2 SSD sister at the canonical-tri-backend surface)
  * :mod:`tac.substrates._shared.inflate_runtime` — Catalog #205 sister discipline

Cross-references (canonical helpers extended by future consumers):
  * :mod:`tac.substrates.z8_hierarchical_predictive_coding.mamba2_adapter` — Z8
    consumer (sister wave will rewire to consume this helper via MLX backend)
  * :mod:`tac.substrates.time_traveler_l5_z7_mamba2` — Z7 consumer (sister
    wave will rewire similarly)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from tac.framework_agnostic.backend import (
    Backend,
    BackendUnavailableError,
    select_backend,
)

# Re-export sister numpy backend (canonical math truth surface).
from .numpy_backend import (
    Mamba2SSDNumpyState,
    mamba2_ssd_forward_sequence_numpy,
    mamba2_ssd_init_state_numpy,
    mamba2_ssd_step_numpy,
)

# Re-export sister PyTorch backend (canonical training surface for paid GPU).
from .pytorch_backend import (
    Mamba2SSDPyTorchState,
    mamba2_ssd_forward_sequence_pytorch,
    mamba2_ssd_init_state_pytorch,
    mamba2_ssd_step_pytorch,
)

# Re-export sister MLX backend (canonical M5 Max training surface per 8th directive).
from .mlx_backend import (
    MLX_AVAILABLE,
    Mamba2SSDMLXState,
    mamba2_ssd_forward_sequence_mlx,
    mamba2_ssd_init_state_mlx,
    mamba2_ssd_step_mlx,
)

__all__ = [
    # Canonical config
    "Mamba2SSDConfig",
    # Tri-backend dispatch
    "compute_mamba2_ssd_forward_sequence",
    # numpy backend (canonical math truth)
    "Mamba2SSDNumpyState",
    "mamba2_ssd_init_state_numpy",
    "mamba2_ssd_step_numpy",
    "mamba2_ssd_forward_sequence_numpy",
    # PyTorch backend (canonical training surface)
    "Mamba2SSDPyTorchState",
    "mamba2_ssd_init_state_pytorch",
    "mamba2_ssd_step_pytorch",
    "mamba2_ssd_forward_sequence_pytorch",
    # MLX backend (canonical M5 Max training surface)
    "Mamba2SSDMLXState",
    "mamba2_ssd_init_state_mlx",
    "mamba2_ssd_step_mlx",
    "mamba2_ssd_forward_sequence_mlx",
    "MLX_AVAILABLE",
    # Re-export Backend enum for caller convenience
    "Backend",
    "BackendUnavailableError",
    "select_backend",
]


@dataclass(frozen=True)
class Mamba2SSDConfig:
    """Static design-time parameters for the canonical Mamba-2 SSD helper.

    Defaults match parent Z8 design memo + canonical state-spaces/mamba upstream
    Mamba2 defaults at contest scale (per parent design memo §7).

    Args:
        nheads: number of SSD heads (canonical scalar-A-per-head; default 2 for
            contest-scale 600-pair sequences per Z8 binding-contract).
        headdim: per-head feature dimension (default 64 — canonical from
            state-spaces/mamba upstream).
        d_state: SSM state dimension (default 16 — Mamba-2 canonical for
            language uses 128; contest 600-pair sequence uses 16 per CC-9
            CARGO-CULTED-PENDING per parent design memo).
        with_skip_connection: whether the optional D · x_t skip connection is
            present (default True — canonical Mamba-2 SSD per upstream).
    """

    nheads: int = 2
    headdim: int = 64
    d_state: int = 16
    with_skip_connection: bool = True

    @property
    def d_inner(self) -> int:
        """Total inner dimension d_inner = nheads × headdim."""
        return self.nheads * self.headdim

    def __post_init__(self) -> None:
        if self.nheads <= 0:
            raise ValueError(f"nheads must be > 0, got {self.nheads}")
        if self.headdim <= 0:
            raise ValueError(f"headdim must be > 0, got {self.headdim}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be > 0, got {self.d_state}")


def compute_mamba2_ssd_forward_sequence(
    *,
    x_seq: Any,
    A_log: Any,
    B_seq: Any,
    C_seq: Any,
    dt_seq: Any,
    D: Any | None = None,
    initial_state: Any | None = None,
    backend: Backend | None = None,
) -> Tuple[Any, Any]:
    """Tri-backend dispatch for Mamba-2 SSD forward over a length-L sequence.

    Routes to the appropriate backend (numpy / PyTorch / MLX) via the canonical
    :func:`select_backend` cascade per Catalog #205 sister discipline.

    The shape contracts are identical across backends (per the byte-stable
    parity invariant); the difference is the tensor type (numpy / torch /
    mlx.core).

    Args:
        x_seq: input sequence (B, L, nheads, headdim) — backend-specific tensor type.
        A_log: SSD A_log (nheads,) — backend-specific tensor type.
        B_seq: per-step B (B, L, nheads, d_state).
        C_seq: per-step C (B, L, nheads, d_state).
        dt_seq: per-step dt (B, L, nheads) — post-softplus.
        D: optional skip (nheads, headdim).
        initial_state: optional state with backend-matching type.
        backend: explicit :class:`Backend`; defaults to AUTO per platform priority.

    Returns:
        Tuple (final_state, y_seq) with backend-matching types.

    Raises:
        BackendUnavailableError: if backend explicitly requested but not installed.
        ValueError: on shape contract violation (raised by per-backend impl).
    """
    resolved = select_backend(override=backend)
    if resolved is Backend.NUMPY:
        return mamba2_ssd_forward_sequence_numpy(
            x_seq=x_seq, A_log=A_log, B_seq=B_seq, C_seq=C_seq,
            dt_seq=dt_seq, D=D, initial_state=initial_state,
        )
    if resolved is Backend.PYTORCH:
        return mamba2_ssd_forward_sequence_pytorch(
            x_seq=x_seq, A_log=A_log, B_seq=B_seq, C_seq=C_seq,
            dt_seq=dt_seq, D=D, initial_state=initial_state,
        )
    if resolved is Backend.MLX:
        return mamba2_ssd_forward_sequence_mlx(
            x_seq=x_seq, A_log=A_log, B_seq=B_seq, C_seq=C_seq,
            dt_seq=dt_seq, D=D, initial_state=initial_state,
        )
    raise BackendUnavailableError(
        f"Backend.{resolved.name} not supported for Mamba-2 SSD forward; "
        f"this helper supports NUMPY / PYTORCH / MLX only. TINYGRAD support "
        f"is a future sister wave per CLAUDE.md 'Forbidden premature KILL'."
    )
