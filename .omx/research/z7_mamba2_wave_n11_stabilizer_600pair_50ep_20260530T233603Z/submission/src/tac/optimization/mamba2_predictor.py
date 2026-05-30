# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 canonical helper — selective state-space recurrent predictor.

Wrapper around the Mamba-2 selective state-space sequence model
(Dao-Gu 2024, arxiv 2405.21060) exposing canonical signatures matching
Z6's ``FilmConditionedNextFramePredictor`` / ``MultiLayerFilmPredictor``
for drop-in compatibility within the predictive-coding-recurrent
substrate class.

Provenance and design provenance
--------------------------------

- **Parent symposium**: ``.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md``
- **Design memo**: ``.omx/research/z7_mamba2_substrate_design_memo_20260518.md``
- **Research wave**: ``.omx/research/comprehensive_research_wave_20260518.md`` TOP-5 #2 + §3.6 DreamerV3-Mamba convergence.
- **Mamba upstream**: https://github.com/state-spaces/mamba ; arxiv 2312.00752 (Mamba) + 2405.21060 (Mamba-2) [verified-against: research_wave_20260518]
- **DreamerV3 upstream** (Hafner Revision #3 binding cites this lineage): https://github.com/danijar/dreamerv3 ; arxiv 2301.04104

Canonical signature contract (matches Z6 sister)
-------------------------------------------------

``Mamba2Predictor.forward(z_prev, ego_motion)`` returns ``(B, latent_dim)``
predicted ``z_t``. The forward sees a single per-pair timestep; the
recurrent state ``h_{t-1}`` is maintained internally between calls when
``stateful=True`` (Wyner-Ziv implicit side-info channel pattern per
Catalog #311 Ballard verbatim) or reset every pair when
``stateful=False`` (ablation control).

Two operating modes:

1. **Production mode** (``backend="mamba_ssm"``): uses the upstream
   CUDA-kernel ``mamba_ssm`` PyPI package. Requires Linux x86_64 +
   CUDA 11.6+. Fast (sister to GRU per upstream benchmarks at language
   scale; sequence-length-600 empirical gap is CARGO-CULTED-PENDING-VERIFICATION
   per parent design memo §2 CC-1).
2. **Reference mode** (``backend="reference_torch"``): pure-PyTorch
   reference implementation; no CUDA kernels. Works on MPS / CPU. ~10×
   slower than ``mamba_ssm`` but architecturally identical. Used for
   local M5 Max MPS proxy training per parent design memo §13.

The backend auto-selects when ``backend="auto"`` (default): tries
``mamba_ssm`` first, falls back to ``reference_torch`` on import error.

Identity-predictor disambiguator mode (Catalog #125 hook #6 + Z7
symposium Revision #2 same-archive-bytes pattern): when
``identity_predictor=True``, the module returns ``z_prev`` unchanged
with no trainable parameters and no recurrent state. This is the
canonical ablation control sister to Z6's identity_predictor mode.

Catalog #220 + #272 distinguishing-feature contract: the Mamba-2
selective state-space mechanism IS the substrate-distinguishing
primitive; runtime overlay must consume it via autoregressive unroll
across 600 pairs; byte-mutation smoke MUST verify mutations on any
Mamba-2 weight produce measurable downstream frame changes.

Catalog #190 hardware substrate: the auto-backend selection writes a
``backend_active`` attribute consumable for canonical hardware
substrate detection downstream.

[verified-against: src/tac/substrates/time_traveler_l5_z6/architecture.py FilmConditionedNextFramePredictor + MultiLayerFilmPredictor]
[verified-against: .omx/research/z7_mamba2_substrate_design_memo_20260518.md §7 architectural specification]

6-hook wire-in declaration per Catalog #125
-------------------------------------------

1. **Sensitivity-map**: Mamba-2 selective projection gradient norms
   (``A_proj``, ``B_proj``, ``C_proj``) ARE the per-tensor importance
   signal; register ``sensitivity_map.time_traveler_l5_z7_mamba2`` at
   Wave N+1 trainer build.
2. **Pareto constraint**: adds ``mamba2_residual_entropy ≤ ε_residual``
   to convex feasibility region; sister to Z6-v1 hook #2.
3. **Bit-allocator hook**: per-pair residual bit allocation derives
   from Mamba-2 selectivity-matrix amplitude; sister to Z6-v1 hook #3.
4. **Cathedral autopilot dispatch hook**: recipe at
   ``.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml``;
   gated by Catalog #167 smoke-before-full + Catalog #325
   per-substrate symposium evidence.
5. **Continual-learning posterior**: every Z7-Mamba-2 empirical anchor
   seeds posterior via ``posterior_update_locked`` (Catalog #128).
6. **Probe-disambiguator**: identity-predictor mode IS the probe; sister
   to Z6 ``identity_predictor`` and Z7-GRU Revision #2 pattern.

Catalog #313 predecessor probe outcome: NOT APPLICABLE — Z7-Mamba-2
has no prior probe outcome.
"""

from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

# Public API
__all__ = [
    "MAMBA_SSM_AVAILABLE",
    "MAMBA_SSM_BACKEND",
    "REFERENCE_TORCH_BACKEND",
    "SSD_REFERENCE_BACKEND",
    "Mamba2Predictor",
    "Mamba2PredictorConfig",
]


REFERENCE_TORCH_BACKEND = "reference_torch"
"""Backend name: pure-PyTorch reference Mamba-1 (S6) implementation; works on MPS/CPU/CUDA.

Honest classification per Wave 4 Dao-Gu fidelity audit (2026-05-29): this
backend implements the diagonal-A-per-channel S6 cell (Gu & Dao 2023; A_log
shape ``(d_inner, d_state)``), NOT the canonical Mamba-2 SSD scalar-A-per-head
form (Dao & Gu 2024 §4; A_log shape ``(nheads,)``). The name is preserved
per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113 to avoid corrupting
cite-chain on existing canonical equation anchors.
"""

MAMBA_SSM_BACKEND = "mamba_ssm"
"""Backend name: upstream mamba_ssm PyPI package; requires CUDA kernels."""

SSD_REFERENCE_BACKEND = "ssd_reference"
"""Backend name: canonical Mamba-2 SSD reference via :mod:`tac.substrates._shared.mamba2_ssd`.

This backend is the canonical Mamba-2 SSD (Dao & Gu 2024 §4 Structured
State-Space Duality) sister to :data:`REFERENCE_TORCH_BACKEND` (Mamba-1 S6).
The cell math delegates to the canonical tri-backend helper which provides
byte-stable parity across NUMPY / PYTORCH / MLX (verified in
:mod:`tac.substrates._shared.mamba2_ssd.tests.test_mamba2_ssd` — 33 passing).

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
#307 paradigm-vs-implementation classification: this backend is
**mathematically distinct** from :data:`REFERENCE_TORCH_BACKEND` — the SSD
form uses ``A_log`` shape ``(nheads,)`` scalar-per-head broadcast across
``(headdim, d_state)`` whereas the S6 reference uses ``A_log`` shape
``(d_inner, d_state)`` diagonal-per-channel. Switching backends produces
a different forward pass; downstream consumers must register
:mod:`tac.canonical_equations` anchors per Catalog #344 on either backend
separately if they want predictive-vs-empirical residual tracking.

Per CLAUDE.md 8th MLX-first standing directive: this backend's MLX route
is the canonical $0 macOS local-substrate path (Z8 M12a MLX-LOCAL,
Z7-Mamba-2 L2 long-training local), unlocking the "all MLX except that
which cannot be done on MLX → make it doable on MLX" trajectory by giving
the existing Z8 + Z7-Mamba-2 consumers a structural MLX path that did not
previously exist.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L8
+ existing :data:`REFERENCE_TORCH_BACKEND` discipline: this backend
preserves gradient flow through the recurrence (no detach in forward) so
score-aware loss can backprop end-to-end.

Cross-reference: :class:`_CanonicalHelperSSDCell` (this module) +
:func:`tac.substrates._shared.mamba2_ssd.compute_mamba2_ssd_forward_sequence`
(tri-backend dispatch entry point).
"""


def _probe_mamba_ssm_available() -> bool:
    """Detect availability of upstream mamba_ssm package (CUDA kernels required).

    Returns True on Linux x86_64 + CUDA 11.6+ when mamba-ssm PyPI package
    is installed. Returns False on macOS / MPS / CPU-only environments
    per CC-4 HARD-EARNED-PARTIAL classification.
    """
    if platform.system() != "Linux" or not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


MAMBA_SSM_AVAILABLE: bool = _probe_mamba_ssm_available()
"""Boolean: True iff upstream mamba_ssm package is importable in this environment."""


@dataclass(frozen=True)
class Mamba2PredictorConfig:
    """Static design-time parameters for Mamba2Predictor.

    Defaults match parent design memo §7 architectural specification.

    Args:
        latent_dim: per-pair latent dimensionality (matches Z6
            ``latent_dim``; default 24).
        ego_motion_dim: ego-motion projection dim (default 8 matches
            Z6-v1 PoseNet-projection baseline; runtime-configurable per
            Z7 symposium Revision #4).
        d_model: Mamba-2 internal model dimension (default 64;
            sister to GRU hidden_dim=128 halved for parameter parity).
        d_state: Mamba-2 selective state-space dimension (default 16;
            Mamba-2 canonical for language; CC-9 CARGO-CULTED-PENDING
            for dashcam contest 600-pair sequence).
        expand: Mamba-2 expansion factor (default 2; canonical from
            upstream reference).
        d_conv: Mamba-2 conv1d kernel size (default 4; canonical).
        backend: one of ``"auto"`` (default; tries mamba_ssm, falls
            back to reference_torch — Mamba-1 S6), ``"mamba_ssm"`` (CUDA
            only — true Mamba-2 SSD), ``"reference_torch"`` (pure
            PyTorch Mamba-1 S6; MPS/CPU compatible), or ``"ssd_reference"``
            (canonical Mamba-2 SSD via :mod:`tac.substrates._shared.mamba2_ssd`
            tri-backend NUMPY/PYTORCH/MLX). The ``"ssd_reference"``
            backend is the canonical $0 macOS path per CLAUDE.md 8th
            MLX-first standing directive; the ``"reference_torch"``
            backend is preserved for backward compatibility and
            cite-chain on existing equation anchors per CLAUDE.md
            HISTORICAL_PROVENANCE Catalog #110/#113.
        ssd_nheads: number of SSD heads when ``backend="ssd_reference"``;
            default ``None`` derives ``nheads = d_inner // ssd_headdim``
            from ``d_inner`` and ``ssd_headdim``. Used to construct the
            canonical helper's ``Mamba2SSDConfig``. Ignored for
            non-SSD backends.
        ssd_headdim: per-head feature dim when ``backend="ssd_reference"``;
            default 64 (canonical from state-spaces/mamba upstream). Used
            with ``ssd_nheads`` to construct the canonical helper's
            ``Mamba2SSDConfig``. Ignored for non-SSD backends. Must
            satisfy ``d_inner % ssd_headdim == 0`` when ``ssd_nheads``
            is None so the head count is integer.
        stateful: True (default) preserves hidden state across forward
            calls (Wyner-Ziv implicit side-info channel pattern per
            Catalog #311 Ballard verbatim); False resets state every
            pair (ablation control).
        identity_predictor: True returns ``z_prev`` unchanged (no
            learning; Catalog #125 hook #6 + Z7 symposium Revision #2
            disambiguator probe).
    """

    latent_dim: int = 24
    ego_motion_dim: int = 8
    d_model: int = 64
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    backend: Literal["auto", "mamba_ssm", "reference_torch", "ssd_reference"] = "auto"
    ssd_nheads: int | None = None
    ssd_headdim: int = 64
    stateful: bool = True
    identity_predictor: bool = False

    @property
    def d_inner(self) -> int:
        """Mamba-2 inner dimension after expansion."""
        return self.expand * self.d_model

    @property
    def predictor_input_dim(self) -> int:
        """Concat dim for (z_prev, ego_motion) input."""
        return self.latent_dim + self.ego_motion_dim


class _ReferenceMamba2Cell(nn.Module):
    """Pure-PyTorch reference selective state-space cell — Mamba-1 (S6) form.

    Wave 4 fidelity audit 2026-05-29 (per Dao & Gu 2024 arxiv 2405.21060
    + state-spaces/mamba upstream + Goomba Lab Mamba-2 blog series):
    this reference cell implements the **Mamba-1 (S6) selective SSM**
    diagonal A parameterization (A_log shape ``(d_inner, d_state)``),
    NOT the canonical Mamba-2 SSD (Structured State-Space Duality) form
    which uses scalar-A-per-head (A_log shape ``(nheads,)`` broadcast to
    ``(nheads, headdim, d_state)``). The MAMBA_SSM_BACKEND path
    correctly invokes the upstream canonical ``mamba_ssm.modules.mamba2.Mamba2``
    block (true Mamba-2 SSD); this reference path is a documented
    adaptation for MPS/CPU/MLX environments where ``mamba_ssm`` is not
    importable.

    **Documented adaptation rationale per Wave 4 audit + CLAUDE.md
    'Forbidden empirical-claim-without-evidence-tag' (the documented-
    adaptation taxonomy axes 3 + 4 — math + data):**

    1. **Contest scale** (latent_dim=24 / d_model=64 / d_state=16 /
       d_inner=128): at this scale, the Mamba-2 SSD scalar-A-per-head
       form with default headdim=64 yields nheads=2 → A_log has only
       2 scalar parameters. The S6 diagonal form provides 2048 (d_inner
       × d_state) parameters at the same overall cell width, which is
       structurally richer per the input-dependent expressivity argument
       in Gu & Dao 2023 §3.4 (Mamba-1 paper). At the canonical Mamba-2
       SSD language scale (d_state=128, headdim=64, model dim ≥1024),
       the SSD chunk-based matmul speedup amortizes the per-head scalar
       constraint; at our 600-pair dashcam contest scale the speedup is
       unrealizable on MPS/CPU/MLX (sequential per-step recurrence
       dominates).

    2. **MPS/CPU compatibility** (CC-4 HARD-EARNED per the Z7-Mamba-2
       cargo-cult audit `path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`):
       the SSD chunk-based parallel scan requires tensor-core matmul
       kernels that don't exist on MPS / CPU at canonical fidelity.
       This reference path provides sequential per-step state updates
       so the recurrence remains differentiable on macOS for local
       smoke + research-signal training.

    **Mathematical fidelity classification per Catalog #307 + the
    documented-adaptation taxonomy:**

    - **PARADIGM-LEVEL FIDELITY (INTACT)**: selective state-space
      recurrence with input-dependent (B, C, dt) matrices is the
      defining feature of the Mamba family (both Mamba-1 S6 and
      Mamba-2 SSD inherit this from Gu & Dao 2023); preserved.
    - **IMPLEMENTATION-LEVEL ADAPTATION (DOCUMENTED)**: the A_log
      diagonal-per-(d_inner, d_state) parameterization is Mamba-1 (S6),
      not Mamba-2 SSD. Honest naming: the substrate would be more
      accurately called "Z7-Selective-SSM" (S6-family), but the
      "Z7-Mamba-2" identifier is preserved per CLAUDE.md
      HISTORICAL_PROVENANCE Catalog #110/#113 (the 4 canonical equation
      anchors on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`
      cite this exact substrate identifier; renaming would corrupt
      cite-chain).

    Per Catalog #220 + #272 distinguishing-feature contract: the
    selective state-space recurrence IS the substrate-distinguishing
    primitive regardless of S6-vs-SSD choice; both forms are
    architecturally distinct from GRU/LSTM at the substrate-class
    layer (per HNeRV parity L7).

    **Discretization** (matches Dao & Gu 2024 §2.2 + Gu & Dao 2023 §2.2
    zero-order hold for both S6 and SSD): ``A_bar = exp(dt * A)``,
    ``B_bar = dt * B`` (the ``B_bar = (1/A) (A_bar - I) B`` exact ZOH
    integral is approximated by ``dt * B`` per upstream Mamba-1
    implementation; both forms preserve the discrete-time recurrence
    structure ``h_t = A_bar * h_{t-1} + B_bar * x_t``).

    **Reactivation criteria for upgrading to true Mamba-2 SSD reference
    cell** (per CLAUDE.md 'Forbidden premature KILL' + the documented-
    adaptation taxonomy):

    1. mamba_ssm CUDA backend becomes the empirical bottleneck at Modal
       T4/A100 → would need true SSD reference for byte-stable parity.
    2. Operator decides to register `mamba_2_ssd_vs_s6_reference_cell_at_contest_scale_v1`
       as a canonical equation per Catalog #344 + run paired-comparison
       smoke at d_state=128 + headdim=64 → 2-head SSD vs 2048-entry S6
       at the contest scale.
    3. New empirical anchor on `z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`
       shows the reference cell's S6 form materially diverges from
       mamba_ssm CUDA backend's SSD form at training-loss trajectory
       level (`predicted_vs_empirical_residual > 2σ`).

    Forward signature: ``(x_t :: (B, d_model), h_{t-1} :: (B, d_inner, d_state)) -> (y_t :: (B, d_model), h_t :: (B, d_inner, d_state))``.

    [verified-against: Dao & Gu 2024 arxiv 2405.21060 §2.2 + §3 (Mamba-2 SSD)]
    [verified-against: Gu & Dao 2023 arxiv 2312.00752 §2.2 + §3.4 (Mamba-1 S6; this implementation matches)]
    [verified-against: state-spaces/mamba upstream `mamba_ssm.modules.mamba2.Mamba2` (canonical Mamba-2 SSD; production CUDA backend)]
    [verified-against: state-spaces/mamba upstream `mamba_ssm.modules.mamba.Mamba` (canonical Mamba-1 S6; this implementation's architectural sister)]
    [verified-against: .omx/research/wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md (this audit)]
    """

    def __init__(self, *, d_model: int, d_state: int, expand: int, d_conv: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model
        self.d_conv = d_conv

        # Input projection: d_model -> 2 * d_inner (split into x and gate
        # per Mamba-1 §3.5). Canonical Mamba-2 SSD upstream uses a different
        # projection structure [z, x, B, C, dt] (see Wave 4 audit memo
        # `wave_4_z7_mamba_2_dao_gu_fidelity_audit_landed_20260529.md` §3.3);
        # this reference path uses Mamba-1 split-gate for MPS/CPU compatibility.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Selective projection matrices per Gu & Dao 2023 §3.4 (Mamba-1 S6 form):
        #   A_log: (d_inner, d_state) diagonal-per-channel; A = -exp(A_log)
        #     keeps eigenvalues negative for stability (canonical S6 init
        #     `log(1..d_state)` broadcast across d_inner channels).
        #   B_proj, C_proj: input-conditioned via x_inner → (B, d_state).
        #
        # Canonical Mamba-2 SSD form per Dao & Gu 2024 §3 + state-spaces/mamba
        # upstream uses A_log: (nheads,) scalar-per-head broadcast to
        # (nheads, headdim, d_state); see Wave 4 audit §3.1 for the
        # documented-adaptation rationale (at contest d_state=16 +
        # d_inner=128 the S6 form is structurally richer per parameter).
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).expand(self.d_inner, d_state).clone())
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        # Step size projection (dt); softplus-activated per Mamba-1 §3.4.
        # Canonical Mamba-2 SSD adds a learned `dt_bias` initialized via
        # inverse-softplus from log-uniform [dt_min=0.001, dt_max=0.1]
        # samples per upstream; this reference uses standard linear+bias
        # which converges acceptably at contest scale per the L2 stability
        # hardening memo's empirical evidence (Cell 2-3 NaN-FREE 30ep
        # under canonical warmup+decay schedule + A_log clamp [-10, 0]).
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-timestep selective state-space step.

        Args:
            x_t: ``(B, d_model)`` input embedding for this timestep.
            h_prev: ``(B, d_inner, d_state)`` previous hidden state.

        Returns:
            ``(y_t :: (B, d_model), h_t :: (B, d_inner, d_state))``.
        """
        # Input + gate projections
        xz = self.in_proj(x_t)  # (B, 2 * d_inner)
        x_inner, z_gate = xz.chunk(2, dim=-1)  # each (B, d_inner)
        # Selective projection
        dt = torch.nn.functional.softplus(self.dt_proj(x_inner))  # (B, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state) negative eigenvalues
        B = self.B_proj(x_inner)  # (B, d_state)
        C = self.C_proj(x_inner)  # (B, d_state)
        # Discretize via zero-order hold: A_bar = exp(dt * A); B_bar = dt * B
        # A.unsqueeze(0) (1, d_inner, d_state) * dt.unsqueeze(-1) (B, d_inner, 1)
        A_bar = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # (B, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(1)  # (B, d_inner, d_state)
        # State update: h_t = A_bar * h_{t-1} + B_bar * x_inner
        h_t = A_bar * h_prev + B_bar * x_inner.unsqueeze(-1)  # (B, d_inner, d_state)
        # Output: y_t = (h_t * C) sum over d_state, gated by z
        y_inner = (h_t * C.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        y_inner = y_inner * torch.sigmoid(z_gate)  # gating
        y_t = self.out_proj(y_inner)  # (B, d_model)
        return y_t, h_t


class _CanonicalHelperSSDCell(nn.Module):
    """Canonical Mamba-2 SSD cell delegating to :mod:`tac.substrates._shared.mamba2_ssd`.

    This cell is the canonical Mamba-2 SSD (Dao & Gu 2024 §4 Structured
    State-Space Duality) sister to :class:`_ReferenceMamba2Cell` (Mamba-1 S6).
    The recurrence math delegates to the canonical tri-backend helper via
    :func:`tac.substrates._shared.mamba2_ssd.compute_mamba2_ssd_forward_sequence`
    or per-step :func:`mamba2_ssd_step_pytorch` — preserving byte-stable
    parity vs NUMPY reference + MLX impl per the canonical helper's 33-test
    parity discipline.

    Shape contract (vs ``_ReferenceMamba2Cell``)
    --------------------------------------------

    The canonical helper operates on structured shape ``(B, nheads, headdim,
    d_state)`` per Dao & Gu 2024 §4 SSD parametrization. This cell reshapes
    between the canonical Mamba2Predictor convention ``(B, d_inner, d_state)``
    (used by the sister S6 reference cell + ``Mamba2Predictor.step_externalized_state``
    contract) and the SSD ``(B, nheads, headdim, d_state)`` form internally.
    Caller sees the SAME ``(B, d_inner, d_state)`` state contract as the
    S6 reference; the SSD-internal reshape is implementation detail.

    Per CLAUDE.md "Subagent coherence-by-default" + Catalog #290
    canonical-vs-unique decision per layer:

    - ADOPT_CANONICAL_BECAUSE_SERVES: ``tac.substrates._shared.mamba2_ssd``
      tri-backend helper (canonical Mamba-2 SSD math; 33 byte-stable parity
      tests; sister consumer surface for Z8 + Z7-Mamba-2 + Z7-Mamba-2-v2-fresh
      + pact_nerv_mamba per the canonical equation registry at Catalog #344).
    - FORK_BECAUSE_PRINCIPLED_MISMATCH: input/output projections (``in_proj``,
      ``out_proj``, ``dt_proj``, ``B_proj``, ``C_proj``) live in this cell
      (not in the helper) because the canonical helper is the MATHEMATICAL
      CORE (low-level SSD primitive) — the projection wrappers are part of
      the Mamba2Predictor architectural sister layer. This is the same
      pattern the S6 reference cell uses (in_proj + cell + out_proj fold
      lives in ``_ReferenceMamba2Cell.forward``); preserved here.

    Documented adaptations per CLAUDE.md "Forbidden empirical-claim-without-
    evidence-tag" + 5-axis taxonomy:

    1. **Axis 4 (math)**: SSD uses scalar-A-per-head (``A_log`` shape
       ``(nheads,)``) NOT diagonal-per-channel (``(d_inner, d_state)``).
       This is the canonical Mamba-2 SSD form per Dao & Gu 2024 §4. The
       parameter count differs from the S6 reference at the A_log surface
       (nheads vs d_inner × d_state); per-call output shape is identical
       at the ``(B, d_inner, d_state)`` state contract surface.
    2. **Axis 3 (problem space)**: gradient-preserving (no detach in
       forward) per CLAUDE.md "HNeRV / leaderboard-implementation parity
       discipline" L8 eval-roundtrip-aware. Verified via the helper's
       :func:`mamba2_ssd_step_pytorch` impl which uses einsum (autograd-traceable).
    3. **Axis 1 (problem space)**: input projection ``in_proj`` produces
       ``2 * d_inner`` (split into ``x`` and gate per Mamba-1 §3.5 +
       canonical Mamba-2 SSD upstream gate-then-residual structure). The
       gate ``z`` multiplies the SSD cell output via ``sigmoid(z)`` per
       upstream — preserved here.
    4. **Axis 5 (data)**: backend selection routes to the helper's
       NUMPY/PYTORCH/MLX dispatch via the canonical :func:`tac.framework_agnostic.backend.select_backend`
       cascade. Default is platform priority (Darwin ARM64 → MLX; Linux →
       PYTORCH; else NUMPY) per the canonical helper docstring; this cell
       pins PYTORCH so the gradient path is intact for substrate training.

    Tri-backend dispatch availability
    ----------------------------------

    This cell's forward pass uses the PyTorch backend of the canonical
    helper (per the gradient discipline above). The full tri-backend
    contract (NUMPY for portability + MLX for $0 macOS) is available via
    the canonical helper directly for inflate-time / MLX-LOCAL substrate
    paths — those paths construct the canonical helper API directly without
    going through this Mamba2Predictor wrapper.

    [verified-against: tac.substrates._shared.mamba2_ssd canonical helper
    docstring + 33 byte-stable parity tests at commit b2936fb81]
    [verified-against: Dao & Gu 2024 arxiv 2405.21060 §4 SSD parametrization]
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        expand: int,
        d_conv: int,
        nheads: int,
        headdim: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        self.nheads = nheads
        self.headdim = headdim
        if self.nheads * self.headdim != self.d_inner:
            raise ValueError(
                f"SSD parametrization requires nheads*headdim == d_inner; "
                f"got nheads={nheads}, headdim={headdim}, d_inner={self.d_inner} "
                f"(expand={expand} * d_model={d_model}). "
                f"Either set ssd_headdim to divide d_inner evenly or "
                f"pin ssd_nheads explicitly."
            )

        # Input projection: d_model -> 2 * d_inner (split into x and gate
        # per Mamba-1 §3.5 + canonical Mamba-2 SSD upstream pattern).
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Per Dao & Gu 2024 §4 SSD scalar-A-per-head form: A_log shape (nheads,).
        # Canonical init from upstream state-spaces/mamba: log-uniform over
        # [1, d_state] per head. This is a STRUCTURAL difference vs S6 reference
        # (d_inner × d_state diagonal). Stability-wise: bound A = -exp(A_log)
        # to keep eigenvalues negative; init range [log(1), log(d_state)] per
        # canonical upstream.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nheads + 1).float()))
        # SSD per-head input-conditioned projections: x_inner (B, d_inner) →
        # B_t (B, nheads, d_state), C_t (B, nheads, d_state), dt_t (B, nheads).
        # The B_proj + C_proj are linear nheads × d_state outputs per head.
        # We use a single Linear that emits (nheads * d_state) then reshape.
        self.B_proj = nn.Linear(self.d_inner, nheads * d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, nheads * d_state, bias=False)
        # Step size projection: x_inner → dt (B, nheads); softplus-activated.
        self.dt_proj = nn.Linear(self.d_inner, nheads, bias=True)
        # Optional skip connection D per Dao & Gu 2024 §4 (canonical Mamba-2
        # SSD includes D · x_t skip; this matches Mamba2SSDConfig.with_skip_connection=True default).
        self.D = nn.Parameter(torch.zeros(nheads, headdim))
        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-timestep canonical Mamba-2 SSD step via :mod:`tac.substrates._shared.mamba2_ssd`.

        Args:
            x_t: ``(B, d_model)`` input embedding for this timestep.
            h_prev: ``(B, d_inner, d_state)`` previous hidden state — flat
                form for sister-API compatibility with ``_ReferenceMamba2Cell``
                + ``Mamba2Predictor.step_externalized_state``. Reshaped
                internally to SSD-structured ``(B, nheads, headdim, d_state)``.

        Returns:
            ``(y_t :: (B, d_model), h_t :: (B, d_inner, d_state))``. Output
            state shape matches the input convention (flat d_inner × d_state).
        """
        # Lazy import the canonical helper at call time so the rest of this
        # module can be imported without forcing the helper's framework
        # backend probes (notably MLX detection) at module-load time.
        from tac.substrates._shared.mamba2_ssd import (
            Mamba2SSDPyTorchState,
            mamba2_ssd_step_pytorch,
        )

        batch = x_t.shape[0]
        # Input + gate projections
        xz = self.in_proj(x_t)  # (B, 2 * d_inner)
        x_inner, z_gate = xz.chunk(2, dim=-1)  # each (B, d_inner)
        # Reshape x_inner to per-head form (B, nheads, headdim).
        x_per_head = x_inner.reshape(batch, self.nheads, self.headdim)
        # Per-step projections per Dao & Gu 2024 §4.
        B_t = self.B_proj(x_inner).reshape(batch, self.nheads, self.d_state)
        C_t = self.C_proj(x_inner).reshape(batch, self.nheads, self.d_state)
        dt_t = torch.nn.functional.softplus(self.dt_proj(x_inner))  # (B, nheads)
        # Reshape flat h_prev (B, d_inner, d_state) -> structured (B, nheads, headdim, d_state).
        h_structured = h_prev.reshape(batch, self.nheads, self.headdim, self.d_state)
        # Delegate to canonical helper PyTorch backend (gradient-preserving).
        next_state, y_per_head = mamba2_ssd_step_pytorch(
            state=Mamba2SSDPyTorchState(h=h_structured),
            x_t=x_per_head,
            A_log=self.A_log,
            B_t=B_t,
            C_t=C_t,
            dt_t=dt_t,
            D=self.D,
        )
        # y_per_head shape: (B, nheads, headdim); flatten to (B, d_inner).
        y_inner = y_per_head.reshape(batch, self.d_inner)
        # Gate y_inner with sigmoid(z_gate) per upstream Mamba-1/Mamba-2 pattern.
        y_inner = y_inner * torch.sigmoid(z_gate)
        # Output projection -> (B, d_model)
        y_t = self.out_proj(y_inner)
        # Flatten next_state shape: (B, nheads, headdim, d_state) -> (B, d_inner, d_state).
        h_t = next_state.h.reshape(batch, self.d_inner, self.d_state)
        return y_t, h_t


class Mamba2Predictor(nn.Module):
    """Z7-Mamba-2 substrate-distinguishing primitive: selective state-space next-frame predictor.

    Per parent design memo §7: predicts ``z_t`` from ``(z_{t-1}, ego_motion[t])``
    via Mamba-2 selective state-space recurrence. The recurrent hidden state
    is maintained internally between forward calls when ``stateful=True``
    (canonical Wyner-Ziv implicit side-info channel pattern per Catalog
    #311 Ballard verbatim).

    Architecture::

        z_prev (B, latent_dim) + ego_motion (B, ego_motion_dim)
                |
                v [concat: (B, latent_dim + ego_motion_dim)]
                |
                v [input_projection: Linear -> (B, d_model)]
                |
                v [Mamba-2 block (selective state-space): (x_t, h_{t-1}) -> (y_t, h_t)]
                |
                v [output_projection: Linear -> (B, latent_dim)]
                |
        z_pred (B, latent_dim)

    Mode ``identity_predictor=True``: returns ``z_prev`` unchanged with
    no trainable parameters; canonical ablation control per Catalog #125
    hook #6 + Z7 symposium Revision #2 same-archive-bytes pattern.

    State management
    ----------------

    When ``stateful=True``, the predictor maintains ``self._h`` as a
    per-batch hidden state across forward calls (the Wyner-Ziv implicit
    side-info channel). Call ``self.reset_state(batch_size, device)``
    at the start of each 600-pair sequence to zero the state. The
    inflate-time deterministic unroll regenerates the same state.

    When ``stateful=False``, each forward call resets state to zero
    (ablation: no temporal coherence; Mamba-2 reduces to a stateless
    nonlinear transform per pair).

    Backend selection
    -----------------

    - ``"auto"``: tries ``mamba_ssm`` (CUDA); falls back to
      ``"reference_torch"`` on ImportError. ``self.backend_active``
      attribute records the actual backend.
    - ``"mamba_ssm"``: forces upstream CUDA kernel; raises ImportError
      on macOS / MPS / CPU.
    - ``"reference_torch"``: pure-PyTorch reference; ~10× slower than
      mamba_ssm at language scale but MPS/CPU compatible per parent
      design memo §13 local M5 Max proxy training pattern.

    [verified-against: parent design memo §7 architectural spec]
    [verified-against: Z7 symposium Revision #3 binding Hafner DreamerV3
    deterministic-recurrence lineage]
    """

    def __init__(self, config: Mamba2PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.ego_motion_dim = config.ego_motion_dim
        self.identity_predictor = config.identity_predictor
        self.stateful = config.stateful

        # Cached hidden state per batch (set on first forward or reset_state)
        self._h: torch.Tensor | None = None

        if config.identity_predictor:
            # No trainable parameters in identity mode
            self.backend_active = "identity"
            return

        # Backend selection
        if config.backend == "mamba_ssm":
            if not MAMBA_SSM_AVAILABLE:
                raise ImportError(
                    "mamba_ssm backend requested but mamba_ssm not importable. "
                    "Install via `pip install mamba-ssm` on Linux x86_64 + CUDA 11.6+. "
                    "macOS / MPS users: use backend='reference_torch', "
                    "backend='ssd_reference', or backend='auto' per Mamba2Predictor "
                    "parent design memo §13."
                )
            self.backend_active = MAMBA_SSM_BACKEND
        elif config.backend == "reference_torch":
            self.backend_active = REFERENCE_TORCH_BACKEND
        elif config.backend == "ssd_reference":
            # Canonical Mamba-2 SSD via tac.substrates._shared.mamba2_ssd
            # tri-backend helper. Per CLAUDE.md 8th MLX-first standing
            # directive + Catalog #344 canonical equation
            # mamba2_ssd_mlx_pytorch_numpy_tri_backend_byte_stable_v1: this
            # is the canonical Mamba-2 SSD reference path (sister to
            # reference_torch which is Mamba-1 S6 reference per Wave 4
            # Dao-Gu fidelity audit 2026-05-29).
            self.backend_active = SSD_REFERENCE_BACKEND
        elif config.backend == "auto":
            self.backend_active = (
                MAMBA_SSM_BACKEND if MAMBA_SSM_AVAILABLE else REFERENCE_TORCH_BACKEND
            )
            if self.backend_active == REFERENCE_TORCH_BACKEND:
                warnings.warn(
                    "Mamba2Predictor: mamba_ssm not available; falling back to "
                    "reference_torch backend (~10x slower at language scale; MPS/CPU "
                    "compatible). Per parent design memo §13 + CC-4 HARD-EARNED-PARTIAL: "
                    "this is canonical for local M5 Max proxy training. For paid Modal "
                    "A100 dispatch, install mamba-ssm explicitly in the training image. "
                    "For canonical Mamba-2 SSD math (true Dao & Gu 2024 §4 SSD form, "
                    "not Mamba-1 S6), pass backend='ssd_reference' to consume "
                    "tac.substrates._shared.mamba2_ssd tri-backend helper.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            raise ValueError(
                f"Unknown backend {config.backend!r}; expected one of "
                "'auto', 'mamba_ssm', 'reference_torch', 'ssd_reference'."
            )
        if self.backend_active == MAMBA_SSM_BACKEND and self.stateful:
            raise RuntimeError(
                "Mamba2Predictor refuses stateful mamba_ssm single-step mode: "
                "the current upstream Mamba2 forward path is length-1 and does "
                "not preserve an incremental inference state. Use "
                "backend='reference_torch' for byte-faithful recurrent evidence, "
                "backend='ssd_reference' for canonical Mamba-2 SSD per Dao & Gu "
                "2024 §4 (via tac.substrates._shared.mamba2_ssd), "
                "set stateful=False for a stateless ablation, or implement "
                "mamba_ssm step/inference-state replay before handoff."
            )

        # Input projection: (latent_dim + ego_motion_dim) -> d_model
        self.input_projection = nn.Linear(
            config.predictor_input_dim, config.d_model, bias=True
        )

        # Mamba-2 cell (backend-specific construction)
        if self.backend_active == MAMBA_SSM_BACKEND:
            # mamba_ssm.modules.mamba2.Mamba2 is the canonical upstream block.
            # We construct a single-cell wrapper since we operate one timestep
            # at a time (consistent with Z6 per-pair forward signature).
            try:
                from mamba_ssm.modules.mamba2 import Mamba2
                self.mamba_cell = Mamba2(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
            except Exception as e:
                # Fall back gracefully if mamba_ssm import partial; keep the
                # backend record honest.
                warnings.warn(
                    f"mamba_ssm.modules.mamba2.Mamba2 construction failed ({e}); "
                    "falling back to reference_torch backend.",
                    UserWarning,
                    stacklevel=2,
                )
                self.backend_active = REFERENCE_TORCH_BACKEND
                self.mamba_cell = _ReferenceMamba2Cell(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    expand=config.expand,
                    d_conv=config.d_conv,
                )
        elif self.backend_active == SSD_REFERENCE_BACKEND:
            # Canonical Mamba-2 SSD cell delegating to canonical helper at
            # tac.substrates._shared.mamba2_ssd (verified byte-stable across
            # NUMPY/PYTORCH/MLX via 33 parity tests at commit b2936fb81).
            #
            # Derive nheads from ssd_nheads/ssd_headdim/d_inner per the
            # SSD parametrization constraint nheads * headdim == d_inner.
            d_inner_local = config.expand * config.d_model
            if config.ssd_nheads is not None:
                nheads_local = config.ssd_nheads
                # Caller-pinned nheads: derive headdim s.t. nheads*headdim == d_inner.
                if d_inner_local % nheads_local != 0:
                    raise ValueError(
                        f"ssd_nheads={nheads_local} does not divide d_inner="
                        f"{d_inner_local} (=expand={config.expand} * d_model="
                        f"{config.d_model}); cannot satisfy nheads*headdim "
                        f"== d_inner SSD constraint."
                    )
                headdim_local = d_inner_local // nheads_local
            else:
                # Default ssd_headdim=64 with auto-derived nheads.
                headdim_local = config.ssd_headdim
                if d_inner_local % headdim_local != 0:
                    raise ValueError(
                        f"ssd_headdim={headdim_local} does not divide d_inner="
                        f"{d_inner_local} (=expand={config.expand} * d_model="
                        f"{config.d_model}); cannot satisfy nheads*headdim "
                        f"== d_inner SSD constraint. Set ssd_nheads or "
                        f"ssd_headdim explicitly to satisfy the constraint."
                    )
                nheads_local = d_inner_local // headdim_local
            self.mamba_cell = _CanonicalHelperSSDCell(
                d_model=config.d_model,
                d_state=config.d_state,
                expand=config.expand,
                d_conv=config.d_conv,
                nheads=nheads_local,
                headdim=headdim_local,
            )
        else:
            self.mamba_cell = _ReferenceMamba2Cell(
                d_model=config.d_model,
                d_state=config.d_state,
                expand=config.expand,
                d_conv=config.d_conv,
            )

        # Output projection: d_model -> latent_dim
        self.output_projection = nn.Linear(
            config.d_model, config.latent_dim, bias=True
        )

    def reset_state(self, batch_size: int, device: torch.device | str = "cpu") -> None:
        """Reset the recurrent hidden state to zeros.

        Must be called at the start of each 600-pair sequence to begin
        the autoregressive unroll. The inflate-time runtime calls this
        once per video.

        Args:
            batch_size: number of parallel sequences.
            device: device for hidden state allocation.
        """
        if self.identity_predictor:
            return
        try:
            dtype = next(self.parameters()).dtype
        except StopIteration:  # pragma: no cover - identity mode returns above
            dtype = torch.float32
        self._h = torch.zeros(
            batch_size,
            self.config.d_inner,
            self.config.d_state,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        z_prev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_t from z_{t-1} + ego_motion via Mamba-2 selective state-space step.

        Args:
            z_prev: ``(B, latent_dim)``.
            ego_motion: ``(B, ego_motion_dim)``.

        Returns:
            ``(B, latent_dim)`` predicted z_t.
        """
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        if self.identity_predictor:
            return z_prev

        batch = z_prev.shape[0]
        # Initialize state if not yet set (or stateful=False)
        if (
            self._h is None
            or not self.stateful
            or self._h.device != z_prev.device
            or self._h.dtype != z_prev.dtype
        ):
            self.reset_state(batch, device=z_prev.device)
        elif self._h.shape[0] != batch:
            # Batch size changed mid-sequence (shouldn't happen in
            # canonical 600-pair unroll); reset to avoid mis-broadcast
            self.reset_state(batch, device=z_prev.device)

        # Concat (z_prev, ego_motion) -> (B, latent_dim + ego_motion_dim)
        predictor_input = torch.cat([z_prev, ego_motion], dim=-1)
        # Input projection -> (B, d_model)
        x_t = self.input_projection(predictor_input)

        # Mamba-2 cell step
        if self.backend_active == MAMBA_SSM_BACKEND:
            # mamba_ssm.Mamba2 operates on (B, L, d_model) sequence;
            # for single-timestep step we unsqueeze and squeeze.
            x_t_seq = x_t.unsqueeze(1)  # (B, 1, d_model)
            # Upstream Mamba2 forward returns (B, L, d_model); for true
            # incremental state we would use Mamba2.step(); the reference
            # path below handles single-step incrementality canonically.
            y_t_seq = self.mamba_cell(x_t_seq)  # (B, 1, d_model)
            y_t = y_t_seq.squeeze(1)  # (B, d_model)
            # Note: state management for mamba_ssm Mamba2 in single-step
            # mode requires `inference_params` from mamba_ssm.utils; for
            # canonical 600-pair training-time unroll we rely on the
            # sequence-mode forward via the trainer (handled in Wave N+1
            # trainer build).
        else:
            # reference_torch backend (Mamba-1 S6) OR ssd_reference backend
            # (canonical Mamba-2 SSD via tac.substrates._shared.mamba2_ssd):
            # both expose the same (x_t, h_prev) -> (y_t, h_t) contract at
            # the flat (B, d_inner, d_state) state shape, so the call site
            # is unified. The cell-specific math distinction (S6 diagonal-A
            # vs SSD scalar-A-per-head) lives inside the cell implementation
            # per Catalog #290 canonical-vs-unique-decision-per-layer.
            y_t, h_t = self.mamba_cell(x_t, self._h)
            if self.stateful:
                self._h = h_t.detach() if not self.training else h_t

        # Output projection -> (B, latent_dim)
        z_pred = self.output_projection(y_t)
        return z_pred

    def step_externalized_state(
        self,
        prior_state: torch.Tensor,
        z_prev: torch.Tensor,
        ego_motion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrence with externalized state pass-through.

        Sister to ``forward`` but does NOT touch ``self._h``. The caller
        passes the recurrent state in explicitly and receives the updated
        state back. This is the canonical interface required by sister
        consumers that want explicit state management — for example the
        Z8 hierarchical predictive coding substrate's
        ``DeterministicStateUpdate`` Protocol per-level recurrence
        (``src/tac/substrates/z8_hierarchical_predictive_coding/
        binding_contract.py``) and any DP1/Z7-Mamba-2 trainer that wants
        deterministic checkpointable hidden-state flow.

        Mathematical grounding
        ----------------------

        The reference cell implements the diagonal-A Mamba-1 S6 selective
        SSM (per Wave 4 fidelity audit landed 2026-05-29 + parent design
        memo §7 + ``_ReferenceMamba2Cell`` docstring). The hidden state
        ``h`` has shape ``(B, d_inner, d_state)`` per Dao & Gu 2024 §3
        (Mamba-1) or ``(B, nheads, headdim, d_state)`` per §4 SSD form
        when the upstream ``mamba_ssm.Mamba2`` backend is active. The
        externalized state shape matches whichever backend is active;
        callers that need shape uniformity should flatten via
        ``state.reshape(B, -1)`` and reshape back at step boundaries.

        Args:
            prior_state: ``(B, d_inner, d_state)`` for the reference
                cell; matching SSD shape for the upstream backend.
            z_prev: ``(B, latent_dim)``.
            ego_motion: ``(B, ego_motion_dim)``.

        Returns:
            ``(next_state, predicted_z)`` where ``next_state`` has the
            same shape as ``prior_state`` and ``predicted_z`` has shape
            ``(B, latent_dim)``.

        Raises:
            RuntimeError: in identity_predictor mode (the identity
                predictor is stateless by construction; the caller must
                handle that case explicitly per Catalog #125 hook #6
                disambiguator semantics).
            NotImplementedError: when the upstream ``mamba_ssm.Mamba2``
                backend is active. The upstream Mamba2 ``forward`` does
                not expose incremental state via a public single-step
                signature; canonical externalization requires
                ``mamba_ssm.utils.inference_params``. Use
                ``backend='reference_torch'`` (Mamba-1 S6) or
                ``backend='ssd_reference'`` (canonical Mamba-2 SSD via
                :mod:`tac.substrates._shared.mamba2_ssd` tri-backend
                helper) for externalized-state consumers per parent
                design memo §13.

        Note:
            The ``ssd_reference`` backend uses the same flat-state
            ``(B, d_inner, d_state)`` contract as the ``reference_torch``
            backend (the SSD-internal reshape to ``(B, nheads, headdim,
            d_state)`` is handled inside :class:`_CanonicalHelperSSDCell`).
            So this externalized-state interface is uniform across
            ``reference_torch`` and ``ssd_reference`` from the caller's
            perspective.
        """
        if self.identity_predictor:
            raise RuntimeError(
                "Mamba2Predictor.step_externalized_state called on identity_predictor "
                "mode; the identity predictor is stateless by construction. The caller "
                "must handle identity mode explicitly (return z_prev, no state update) "
                "per Catalog #125 hook #6 disambiguator semantics."
            )
        if self.backend_active == MAMBA_SSM_BACKEND:
            raise NotImplementedError(
                "Mamba2Predictor.step_externalized_state currently supports only "
                "backend='reference_torch' (Mamba-1 S6) or backend='ssd_reference' "
                "(canonical Mamba-2 SSD via tac.substrates._shared.mamba2_ssd). "
                "The upstream mamba_ssm.Mamba2 sequence-mode forward does not "
                "expose incremental state via a stable single-step signature; "
                "canonical externalization requires mamba_ssm.utils."
                "inference_params (per parent design memo §13). For externalized-state "
                "consumers (Z8 DeterministicStateUpdate, DP1, etc.) use "
                "backend='reference_torch' or backend='ssd_reference'."
            )
        if z_prev.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z_prev last dim {z_prev.shape[-1]} != latent_dim {self.latent_dim}"
            )
        if ego_motion.shape[-1] != self.ego_motion_dim:
            raise ValueError(
                f"ego_motion last dim {ego_motion.shape[-1]} != ego_motion_dim "
                f"{self.ego_motion_dim}"
            )
        expected_h_shape = (z_prev.shape[0], self.config.d_inner, self.config.d_state)
        if tuple(prior_state.shape) != expected_h_shape:
            raise ValueError(
                f"prior_state shape {tuple(prior_state.shape)} != expected "
                f"{expected_h_shape} (B, d_inner, d_state)"
            )

        # Pure functional step — does NOT mutate self._h.
        predictor_input = torch.cat([z_prev, ego_motion], dim=-1)
        x_t = self.input_projection(predictor_input)
        y_t, next_state = self.mamba_cell(x_t, prior_state)
        z_pred = self.output_projection(y_t)
        return next_state, z_pred

    def num_parameters(self) -> int:
        """Count of trainable parameters (sister signature to Z6 predictors)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_z6_compatible_signature(self) -> str:
        """Return human-readable string confirming canonical signature match with Z6.

        Returns: ``"Mamba2Predictor canonical signature: forward(z_prev: (B,
        latent_dim), ego_motion: (B, ego_motion_dim)) -> (B, latent_dim)"``.
        Matches Z6 ``FilmConditionedNextFramePredictor.forward`` and
        ``MultiLayerFilmPredictor.forward`` per design memo §6 layer #2.
        """
        return (
            f"Mamba2Predictor canonical signature: forward(z_prev: "
            f"(B, {self.latent_dim}), ego_motion: (B, {self.ego_motion_dim})) "
            f"-> (B, {self.latent_dim}); backend_active={self.backend_active!r}; "
            f"stateful={self.stateful}; identity_predictor={self.identity_predictor}"
        )


# Catalog #810 per-pair master gradient compatibility note:
# Per-pair master gradient passes z_prev as a leaf tensor with requires_grad=True
# and propagates ∂L/∂z_prev. Mamba2Predictor's forward IS differentiable
# w.r.t. z_prev (the input concat preserves the gradient path) and w.r.t.
# ego_motion (the input projection preserves the gradient path), assuming
# the underlying mamba_ssm.Mamba2 or _ReferenceMamba2Cell is differentiable.
# The reference cell is fully differentiable (no detach in the forward path
# except the optional state-cache detach when not training).
# The upstream mamba_ssm CUDA kernel is differentiable per the upstream
# README and test suite; verified in Wave N+1 trainer build.
