# SPDX-License-Identifier: MIT
"""Z8 Phase 2 M4 — Mamba-2 adapter binding ``tac.optimization.mamba2_predictor``
to the per-level ``DeterministicStateUpdate`` Protocol.

This adapter is the canonical Z8 binding for milestone M4
(``mamba_2_adapter_binds_canonical_primitive_to_protocol``). It does NOT reimplement the
Mamba-2 selective state-space recurrence — it WRAPS the existing
canonical primitive ``tac.optimization.mamba2_predictor.Mamba2Predictor``
to satisfy the per-level binding contract declared in
``binding_contract.DeterministicStateUpdate``.

The binding-first methodology
-----------------------------

Per the Z8 binding-first methodology (sister memory
``z8-hierarchical-predictive-coding-binding-first-active-build-target-
yousfi-grounded-20260529``): each Z8 forward-pass primitive is BOUND
to the contract — not RE-implemented inside the substrate. This adapter
is the canonical worked example:

1. Reuse the canonical primitive (``Mamba2Predictor`` — landed +
   Wave 4 Dao-Gu fidelity audit verified 2026-05-29).
2. Reuse the canonical externalized-state surface
   (``Mamba2Predictor.step_externalized_state`` — landed sister commit
   per operator's "iterate underlying pieces as well" directive
   2026-05-29).
3. Wrap with thin shape-adapter to satisfy the per-level
   ``DeterministicStateUpdate`` Protocol's ``(B, state_dim)`` shape
   contract.

Mathematical grounding
----------------------

The Z8 ``DeterministicStateUpdate`` Protocol declares
``state_dim: int`` and shape contract ``(batch_size, state_dim)``. The
underlying ``Mamba2Predictor.step_externalized_state`` operates on
structured state ``(B, d_inner, d_state)`` per Dao & Gu 2024 §3-§4
(d_inner = expand × d_model; d_state = selective-scan state dimension).
This adapter:

- Maps the level's contracted ``deterministic_state_dim`` to the
  product ``d_inner × d_state`` of the underlying ``Mamba2PredictorConfig``.
  The product MUST equal the level's contracted state_dim.
- Reshapes externally between ``(B, state_dim)`` flat (Protocol) and
  ``(B, d_inner, d_state)`` structured (underlying primitive) at step
  boundaries. The reshape is byte-stable and gradient-preserving (a
  view, not a copy, when possible).
- Per-level ``input_dim`` (the dimension of ``input_at_t``) is
  ``latent_dim + ego_motion_dim`` per the Mamba2Predictor input
  projection. Z8's per-level input is the concatenation of the prior
  level's pose projection + this level's latent prediction; the
  ``input_dim`` matches ``Mamba2PredictorConfig.predictor_input_dim``.

Honest classification
---------------------

The reference backend (``backend='reference_torch'`` /
``REFERENCE_TORCH_BACKEND``) implements Mamba-1 S6 (per Wave 4 fidelity
audit landed 2026-05-29 + canonical helper docstring). The upstream
``mamba_ssm.Mamba2`` (``MAMBA_SSM_BACKEND``) is the canonical Mamba-2
SSD form per Dao & Gu 2024 §4 but is CUDA-only and does NOT currently
expose a public externalized-state single-step signature.

**2026-05-30 canonical SSD reference upgrade**: a third backend
``backend='ssd_reference'`` (``SSD_REFERENCE_BACKEND``) is now
available which routes through the canonical tri-backend helper at
:mod:`tac.substrates._shared.mamba2_ssd` (per commit ``b2936fb81``;
33 byte-stable parity tests passing across NUMPY/PYTORCH/MLX). This
backend implements the canonical Mamba-2 SSD scalar-A-per-head form
(Dao & Gu 2024 §4) — mathematically distinct from
``reference_torch`` (Mamba-1 S6 diagonal-A-per-channel per Gu & Dao
2023). The new ``use_canonical_ssd=True`` adapter kwarg opts into
this backend; the default remains ``use_canonical_ssd=False`` so
existing Z8 milestone evidence + canonical equation anchors continue
to cite the reference-torch backend unchanged per CLAUDE.md
HISTORICAL_PROVENANCE Catalog #110/#113 + Catalog #344 cite-chain
discipline.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this
docstring claims NO score improvement, NO empirical benchmark, NO
contest-axis result. It claims one thing: structural Protocol
satisfaction with mathematically-honest backend classification. Score
claims belong to downstream M11 (L1 macOS-CPU smoke) + M12 (paired-CUDA
sub-0.189 threshold) per ``build_progress.Z8_PHASE_2_BUILD_MILESTONES``.

[verified-against: tac.optimization.mamba2_predictor.Mamba2Predictor
docstring + Wave 4 Dao-Gu fidelity audit landed 2026-05-29]
[verified-against: tac.substrates._shared.mamba2_ssd canonical helper
docstring + 33 byte-stable parity tests at commit b2936fb81]
[verified-against:
src/tac/substrates/z8_hierarchical_predictive_coding/binding_contract.py
DeterministicStateUpdate Protocol]

Per Catalog #290 canonical-vs-unique decision per layer:

- ADOPT_CANONICAL_BECAUSE_SERVES: ``tac.optimization.mamba2_predictor``
  (the canonical Mamba-2 primitive; sister surface for Z7-Mamba-2 +
  DP1 + future predictive-recurrence substrates). When
  ``use_canonical_ssd=True``, also ADOPT_CANONICAL_BECAUSE_SERVES on
  ``tac.substrates._shared.mamba2_ssd`` (the canonical tri-backend SSD
  helper providing $0 macOS MLX path per CLAUDE.md 8th MLX-first
  standing directive).
- FORK_BECAUSE_PRINCIPLED_MISMATCH: NOT applicable at this layer.
  The Z8 binding contract is satisfied by direct wrap; no fork is
  needed at the adapter surface.
"""

from __future__ import annotations

from typing import Any

import torch

from tac.optimization.mamba2_predictor import (
    REFERENCE_TORCH_BACKEND,
    SSD_REFERENCE_BACKEND,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)

from .binding_contract import (
    LevelDimensionContract,
)

__all__ = [
    "Z8Mamba2DeterministicStateUpdate",
    "build_z8_mamba2_adapter_for_level",
]


class Z8Mamba2DeterministicStateUpdate:
    """Z8 per-level ``DeterministicStateUpdate`` adapter wrapping ``Mamba2Predictor``.

    Satisfies the Protocol declared in ``binding_contract.
    DeterministicStateUpdate``: ``state_dim`` property, ``initial_state(batch_size)``
    method, ``step(prior_state, input_at_t)`` method.

    The instance pins ``backend='reference_torch'`` per the honest-
    classification discipline in the module docstring above. Upgrading
    to true Mamba-2 SSD requires the upstream ``mamba_ssm.utils.
    inference_params`` integration (deferred to milestone M4's stronger
    acceptance criterion).

    Args:
        level: the ``LevelDimensionContract`` for the Z8 level this
            adapter binds to. The adapter's ``state_dim`` equals
            ``level.deterministic_state_dim``. The underlying
            ``Mamba2PredictorConfig`` is constructed so that
            ``d_inner × d_state == level.deterministic_state_dim``.
        latent_dim: the per-level latent dimension (passes through to
            ``Mamba2PredictorConfig.latent_dim``; used by the
            ``input_projection`` of the underlying primitive).
        ego_motion_dim: the per-level ego-motion projection dim
            (passes through to ``Mamba2PredictorConfig.ego_motion_dim``).
            For Z8, the ``input_at_t`` passed to ``step`` is the
            concatenation of (prior level pose projection + this
            level's z_prev). The total length MUST equal
            ``latent_dim + ego_motion_dim`` per
            ``Mamba2PredictorConfig.predictor_input_dim``.
        d_state: the selective-scan state dimension. Defaults to 16
            (Mamba-2 canonical for language; CC-9 CARGO-CULTED-PENDING
            for dashcam contest 600-pair sequence per parent design
            memo §7 + Wave 4 audit). ``d_inner`` is derived to satisfy
            ``d_inner × d_state == level.deterministic_state_dim``.
        use_canonical_ssd: opt-in to the canonical Mamba-2 SSD reference
            backend at :mod:`tac.substrates._shared.mamba2_ssd` (true
            Dao & Gu 2024 §4 scalar-A-per-head SSD form via the tri-
            backend NUMPY/PYTORCH/MLX helper; verified byte-stable at
            commit ``b2936fb81``). Default ``False`` preserves the
            Mamba-1 S6 reference backend so existing Z8 milestone
            evidence + canonical equation anchors continue to cite the
            existing backend unchanged. When ``True``, the adapter
            constructs ``Mamba2PredictorConfig(backend='ssd_reference',
            ssd_nheads=1, ssd_headdim=d_inner)`` so the SSD parametrization
            is satisfied at the d_inner=d_inner_local pinned by the
            level contract (nheads=1 keeps the per-head scalar-A
            parameter to a single value at this small contest scale,
            matching the Z8 binding-contract per-level state_dim
            semantics). Per CLAUDE.md 8th MLX-first standing directive:
            this opt-in unlocks the $0 macOS MLX-LOCAL path that
            previously did not exist for Z8 M12a paid Modal T4 dispatch.
        ssd_nheads: optional override for the SSD nheads parameter when
            ``use_canonical_ssd=True``. Default ``None`` uses
            ``nheads=1`` (single-head SSD at the per-level state_dim
            scale; preserves Protocol contract). Setting >1 requires
            ``d_inner % nheads == 0`` per the SSD parametrization
            constraint. Ignored when ``use_canonical_ssd=False``.

    Raises:
        ValueError: if the level's ``deterministic_state_dim`` is not
            evenly divisible by ``d_state`` (the structured state
            shape ``(B, d_inner, d_state)`` requires integer
            ``d_inner``); or if ``ssd_nheads`` does not divide
            ``d_inner`` when ``use_canonical_ssd=True``.
    """

    def __init__(
        self,
        level: LevelDimensionContract,
        latent_dim: int,
        ego_motion_dim: int,
        d_state: int = 16,
        *,
        use_canonical_ssd: bool = False,
        ssd_nheads: int | None = None,
    ) -> None:
        if not isinstance(level, LevelDimensionContract):
            raise TypeError(
                f"level must be LevelDimensionContract, got {type(level).__name__}"
            )
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if ego_motion_dim <= 0:
            raise ValueError(f"ego_motion_dim must be > 0, got {ego_motion_dim}")
        if d_state <= 0:
            raise ValueError(f"d_state must be > 0, got {d_state}")
        if level.deterministic_state_dim % d_state != 0:
            raise ValueError(
                f"level.deterministic_state_dim={level.deterministic_state_dim} "
                f"is not divisible by d_state={d_state}; the structured Mamba-2 "
                f"state shape (B, d_inner, d_state) requires integer d_inner. "
                f"Either choose a different d_state (must divide "
                f"{level.deterministic_state_dim}) or adjust the level's "
                f"deterministic_state_dim contract."
            )

        self._level = level
        self._d_state = d_state
        self._use_canonical_ssd = bool(use_canonical_ssd)
        d_inner = level.deterministic_state_dim // d_state
        # The Mamba2PredictorConfig parameterizes d_inner = expand * d_model.
        # We pin expand=1 and d_model=d_inner so the structured shape
        # contract matches exactly without forcing the caller to know
        # the upstream parameterization. expand=1 is a legitimate
        # configuration per upstream defaults (no constraint requires
        # expand=2; CC-9 in parent design memo flagged expand=2 as
        # CARGO-CULTED-PENDING).
        if self._use_canonical_ssd:
            # Canonical Mamba-2 SSD via tac.substrates._shared.mamba2_ssd.
            # Default ssd_nheads=1 at this small contest scale so the
            # scalar-A-per-head parameter is single-valued; caller can
            # override for multi-head SSD parametrization.
            effective_nheads = ssd_nheads if ssd_nheads is not None else 1
            if d_inner % effective_nheads != 0:
                raise ValueError(
                    f"ssd_nheads={effective_nheads} does not divide d_inner="
                    f"{d_inner}; SSD parametrization requires "
                    f"nheads*headdim == d_inner with integer headdim."
                )
            effective_headdim = d_inner // effective_nheads
            self._config = Mamba2PredictorConfig(
                latent_dim=latent_dim,
                ego_motion_dim=ego_motion_dim,
                d_model=d_inner,
                d_state=d_state,
                expand=1,
                d_conv=4,
                backend=SSD_REFERENCE_BACKEND,
                ssd_nheads=effective_nheads,
                ssd_headdim=effective_headdim,
                stateful=False,
                identity_predictor=False,
            )
        else:
            self._config = Mamba2PredictorConfig(
                latent_dim=latent_dim,
                ego_motion_dim=ego_motion_dim,
                d_model=d_inner,
                d_state=d_state,
                expand=1,
                d_conv=4,
                backend=REFERENCE_TORCH_BACKEND,
                stateful=False,  # externalized-state mode; per-call state pass-through
                identity_predictor=False,
            )
        self._predictor = Mamba2Predictor(self._config)
        # Sanity: confirm d_inner derivation matches what the predictor expects.
        assert self._predictor.config.d_inner == d_inner, (
            f"d_inner mismatch: derived {d_inner} vs predictor "
            f"{self._predictor.config.d_inner}"
        )

    @property
    def state_dim(self) -> int:
        """Per-level deterministic state dimension (flat shape Protocol contract).

        Matches the level's ``LevelDimensionContract.deterministic_state_dim``
        exactly; the underlying Mamba-2 cell uses the structured shape
        ``(B, d_inner, d_state)`` internally with ``d_inner × d_state == state_dim``.
        """
        return self._level.deterministic_state_dim

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """Produce the zero initial state at flat shape ``(batch_size, state_dim)``.

        The state is initialized to zeros per the canonical autoregressive-
        unroll convention (matches ``Mamba2Predictor.reset_state``); the
        inflate-time runtime deterministically regenerates the same zero
        initial state per Catalog #311 Ballard verbatim implicit-side-
        info-channel pattern.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        try:
            dtype = next(self._predictor.parameters()).dtype
            device = next(self._predictor.parameters()).device
        except StopIteration:
            dtype = torch.float32
            device = torch.device("cpu")
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def step(self, prior_state: Any, input_at_t: Any) -> torch.Tensor:
        """Single-step recurrence at flat shape contract.

        Args:
            prior_state: ``(batch_size, state_dim)`` flat. Reshaped
                internally to the underlying ``(B, d_inner, d_state)``
                structured shape.
            input_at_t: ``(batch_size, latent_dim + ego_motion_dim)``.
                Split internally into ``(z_prev, ego_motion)`` per the
                underlying ``Mamba2Predictor.step_externalized_state``
                signature.

        Returns:
            ``(batch_size, state_dim)`` flat next state. The
            predicted-z output of the underlying primitive is NOT
            returned at this surface because the Z8 Protocol declares
            only state in / state out; downstream Z8 substrate code
            that wants the predicted-z output should bypass this
            adapter and call ``self._predictor.step_externalized_state``
            directly.

        Raises:
            ValueError: on shape mismatch.
        """
        if not isinstance(prior_state, torch.Tensor):
            raise TypeError(
                f"prior_state must be torch.Tensor, got {type(prior_state).__name__}"
            )
        if not isinstance(input_at_t, torch.Tensor):
            raise TypeError(
                f"input_at_t must be torch.Tensor, got {type(input_at_t).__name__}"
            )
        if prior_state.dim() != 2 or prior_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"prior_state shape {tuple(prior_state.shape)} expected "
                f"(batch_size, {self.state_dim})"
            )
        expected_input_dim = self._config.predictor_input_dim
        if input_at_t.dim() != 2 or input_at_t.shape[-1] != expected_input_dim:
            raise ValueError(
                f"input_at_t shape {tuple(input_at_t.shape)} expected "
                f"(batch_size, {expected_input_dim}) = "
                f"(batch_size, latent_dim + ego_motion_dim)"
            )
        if prior_state.shape[0] != input_at_t.shape[0]:
            raise ValueError(
                f"batch size mismatch: prior_state {prior_state.shape[0]} vs "
                f"input_at_t {input_at_t.shape[0]}"
            )

        batch = prior_state.shape[0]
        # Reshape flat (B, state_dim) -> structured (B, d_inner, d_state).
        structured_state = prior_state.reshape(
            batch, self._config.d_inner, self._d_state
        )
        # Split input_at_t -> (z_prev, ego_motion) per the underlying primitive.
        latent_dim = self._config.latent_dim
        z_prev = input_at_t[:, :latent_dim]
        ego_motion = input_at_t[:, latent_dim:]
        next_structured, _z_pred = self._predictor.step_externalized_state(
            structured_state, z_prev, ego_motion
        )
        # Reshape structured -> flat for Protocol return.
        return next_structured.reshape(batch, self.state_dim)


def build_z8_mamba2_adapter_for_level(
    level: LevelDimensionContract,
    latent_dim: int,
    ego_motion_dim: int,
    d_state: int = 16,
    *,
    use_canonical_ssd: bool = False,
    ssd_nheads: int | None = None,
) -> Z8Mamba2DeterministicStateUpdate:
    """Convenience constructor (canonical single-line builder per Catalog #290).

    Equivalent to ``Z8Mamba2DeterministicStateUpdate(level, latent_dim,
    ego_motion_dim, d_state=d_state, use_canonical_ssd=use_canonical_ssd,
    ssd_nheads=ssd_nheads)`` but exists as a top-level function so
    downstream consumers can adopt a uniform builder pattern across the
    M4 / M5 / M7 / M8 adapter set without importing the class directly.

    See :class:`Z8Mamba2DeterministicStateUpdate` for the
    ``use_canonical_ssd`` + ``ssd_nheads`` semantics (canonical Mamba-2
    SSD via :mod:`tac.substrates._shared.mamba2_ssd` tri-backend helper
    per CLAUDE.md 8th MLX-first standing directive).
    """
    return Z8Mamba2DeterministicStateUpdate(
        level=level,
        latent_dim=latent_dim,
        ego_motion_dim=ego_motion_dim,
        d_state=d_state,
        use_canonical_ssd=use_canonical_ssd,
        ssd_nheads=ssd_nheads,
    )
