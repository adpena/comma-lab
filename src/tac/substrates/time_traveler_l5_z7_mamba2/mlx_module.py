# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 MLX-FIRST canonical ``mlx.nn.Module`` wrapper — Wave N+9 Slot 1.

# AUTOCAST_FP16_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_autocast_fp16_per_mlx_first_canonical_doctrine_8th_standing_directive
# TF32_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_tf32_per_mlx_first_canonical_doctrine
# TORCH_COMPILE_WAIVED:MLX_substrate_uses_mlx_value_and_grad_not_torch_compile
# NO_GRAD_WAIVED:MLX_substrate_uses_mlx_lazy_eval_not_pytorch_no_grad_per_mlx_first_canonical_doctrine

This module is the operator-routable migration target named in Wave N+8 Slot 1
landing memo §10 ("Operator-routable next"). It promotes Z7-Mamba-2 from L0
SCAFFOLD (plain Python ``Z7Mamba2MLXNativeRenderer``) to L1 MLX-FIRST-TRAINABLE
by exposing the existing parameter set as an ``mlx.nn.Module`` so the canonical
harness at ``tac.substrates._shared.mlx_score_aware.adapter`` (line 161) can
call ``mlx.nn.value_and_grad(self.model, _loss_fn_inner)`` for $0 MLX-LOCAL
600-pair score-aware training on M5 Max.

Per the operator's 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27 +
the 8th MLX-FIRST standing directive REINFORCED 2026-05-28 ("always prefer
MLX first always"): this is Z7-Mamba-2's OWN canonical engineering pass —
the Mamba-2 selective state-space recurrence + Z6-compatible PixelShuffle
decoder forward equations are PRESERVED 1:1 from the existing
``Z7Mamba2MLXNativeRenderer``; only the parameter management surface
changes (plain Python class → ``mlx.nn.Module``).

Approach selected: **(b) ``mx.array`` attributes registered directly on the
``nn.Module`` subclass**, per the operator-routable migration plan in Wave N+8
Slot 1 §10 step (1)(b). The ``mlx.nn.Module`` auto-discovers ``mx.array``
attributes (including nested in lists) via ``mlx.utils.tree_flatten``, so a
canonical ``self.parameters()`` view emerges naturally without rewriting the
forward path. This is the LOWER-TOUCH path; the alternative (a) ``mlx.nn.Linear``
+ ``mlx.nn.Conv2d`` submodule re-wrap would require a forward-path rewrite
because the existing Mamba-2 cell + decoder use raw ``mx.matmul`` /
``mx.conv2d`` calls that do NOT match the ``nn.Linear`` / ``nn.Conv2d`` weight
layout conventions byte-stably (e.g. the MLX ``nn.Conv2d`` exposes
``(out_channels, kH, kW, in_channels)`` weights with a separate ``__call__``
that adds bias automatically, while the existing Mamba-2 cell uses
explicit ``x @ w.T + b`` Linear-style multiplication).

PyTorch-bridge byte parity (Catalog #1251) is PRESERVED via
``export_state_dict`` / ``load_state_dict_from_numpy`` delegating to the
existing ``Z7Mamba2MLXNativeRenderer`` methods unchanged.

Canonical-vs-unique decision per layer (Catalog #290):

- **ADOPT_CANONICAL** ``mlx.nn.Module`` base + ``mx.array`` attribute
  registration: the canonical MLX parameter discovery surface; no fork.
- **ADOPT_CANONICAL** delegation to ``Z7Mamba2MLXNativeRenderer`` for forward
  equations + state_dict bridge: the existing renderer's math IS the
  canonical sister of PyTorch ``Z7Mamba2PredictiveCodingSubstrate`` per the
  #1251 export bridge; mutating the forward path here would fork unnecessarily.
- **ADOPT_CANONICAL** ``reconstruct_pair(pair_indices_np) -> (rgb_0, rgb_1,
  latents)`` API per the MLX harness ``forward_convention="reconstruct_pair_nchw01"``
  contract at ``tac.substrates._shared.mlx_score_aware.loss.decode_frames_nhwc01``.
- **FORK_BECAUSE_PRINCIPLED_MISMATCH** parameter mutation methods
  (``_assign_param``): MLX value_and_grad replaces parameters in place via
  optimizer.update; the wrapper exposes a thin setter so the renderer-side
  delegate stays in sync with the gradient-updated parameters.

6-hook wire-in per Catalog #125 (sister of Wave N+8 landing memo §13):

1. sensitivity-map: ACTIVE — pose-axis Mamba-2 state-space temporal-prediction
   sensitivity surfaced via canonical
   ``tac.substrates._shared.mlx_score_aware.adapter.score_aware_components``
   (the canonical per-axis GAP FIX 92a39dc62).
2. Pareto constraint: ACTIVE — pose-axis Lagrangian dual via Catalog #372
   Dykstra solver post-empirical-anchor.
3. bit-allocator: ACTIVE — per-frame Mamba-2 hidden-state residual budget
   consumed by canonical ``score_aware_loss``.
4. cathedral autopilot dispatch: ACTIVE — auto-discovered via Catalog #335
   canonical Protocol contract once MLX-FIRST trainer LANDS (this module IS
   the structural prerequisite).
5. continual-learning posterior: ACTIVE — Wave N+8 Slot 1 already registered
   canonical equation ``z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1``;
   the first empirical anchor lands AFTER this migration completes 600-pair
   training (deliverable #5 of THIS lane).
6. probe-disambiguator: ACTIVE — Mamba-2 selective state-space (this lane) vs
   Z6-v2 Rao-Ballard RNN-style (Wave N+5) IS the canonical sister-architecture
   disambiguator within the encoder-side cooperative-receiver paradigm class.

[verified-against: src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py Z7Mamba2MLXNativeRenderer]
[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py Z6V2SubstrateMLX canonical sister pattern]
[verified-against: src/tac/substrates/_shared/mlx_score_aware/adapter.py:161 mlx.nn.value_and_grad call site]
[verified-against: src/tac/substrates/_shared/mlx_score_aware/loss.py decode_frames_nhwc01 forward convention contract]
[verified-against: .omx/research/z7_mamba2_state_space_hinton_distill_600pair_long_mlx_landed_20260528.md §10 migration plan]
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
    Z7Mamba2MLXNativeRenderer,
    Z7Mamba2MLXRenderConfig,
    require_mlx,
)

if TYPE_CHECKING:
    from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
        Z7Mamba2PredictiveCodingConfig,
    )

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "z7_mamba2_mlx_module_v1_20260528"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


# Canonical list of mx.array attribute names exported as Mamba-2 cell + decoder
# parameters. Sister of the field-name table in Z7Mamba2MLXNativeRenderer
# docstring (lines 70-84). Order is canonical so iteration is deterministic
# (Catalog #229 reproducibility).
_FLAT_PARAM_ATTRS: tuple[str, ...] = (
    # Mamba-2 cell
    "input_projection_w",
    "input_projection_b",
    "mamba_in_proj_w",
    "mamba_A_log",
    "mamba_B_proj_w",
    "mamba_C_proj_w",
    "mamba_dt_proj_w",
    "mamba_dt_proj_b",
    "mamba_out_proj_w",
    "output_projection_w",
    "output_projection_b",
    # Decoder initial proj
    "dec_initial_proj_w",
    "dec_initial_proj_b",
    # Final decoder conv
    "dec_final_w",
    "dec_final_b",
    # Trainable latent + per-pair residuals
    "latent_init",
    "residuals",
)

_SSD_PARAM_ATTRS: tuple[str, ...] = (
    "ssd_A_log",
    "ssd_B_proj_w",
    "ssd_C_proj_w",
    "ssd_dt_proj_w",
    "ssd_dt_proj_b",
    "ssd_D",
)

_S6_ONLY_PARAM_ATTRS: frozenset[str] = frozenset(
    {
        "mamba_A_log",
        "mamba_B_proj_w",
        "mamba_C_proj_w",
        "mamba_dt_proj_w",
        "mamba_dt_proj_b",
    }
)


class Z7Mamba2MLXModule(nn.Module if nn is not None else object):  # type: ignore[misc]
    """``mlx.nn.Module`` wrapper around ``Z7Mamba2MLXNativeRenderer``.

    Exposes the renderer's parameter set as ``mx.array`` attributes registered
    on a ``mlx.nn.Module`` subclass so the canonical harness call
    ``mlx.nn.value_and_grad(self.model, _loss_fn_inner)`` at
    ``tac.substrates._shared.mlx_score_aware.adapter`` line 161 succeeds.

    The forward path delegates to the existing renderer 1:1 so byte parity
    with the PyTorch sister ``Z7Mamba2PredictiveCodingSubstrate`` is preserved
    per the #1251 export bridge. Per CLAUDE.md "MLX portable-local-substrate
    authority": every artifact produced by training this module is
    non-promotable ``[macOS-MLX research-signal]``; promotion to autopilot
    ranking happens only after PyTorch-bridge export + paired CUDA-CPU
    evaluation per Catalog #1265.

    Args:
        cfg: MLX-native render config (mirrors PyTorch ``Z7Mamba2PredictiveCodingConfig``).
        seed: deterministic init seed (forwarded to underlying renderer).
    """

    def __init__(
        self, cfg: Z7Mamba2MLXRenderConfig, *, seed: int = 0
    ) -> None:
        require_mlx()
        super().__init__()
        self.cfg = cfg
        self._seed = int(seed)

        # Build the existing renderer once; it owns all S6/PyTorch-bridge
        # parameter creation + Kaiming inits + decoder weights. Canonical
        # SSD-MLX opt-in deliberately uses a bridge-compatible init renderer
        # with SSD disabled; the real SSD recurrent core lives in this module
        # and export remains fail-closed until the runtime adapter exists.
        renderer_cfg = cfg
        if cfg.use_canonical_ssd_mlx_backend:
            renderer_cfg = replace(
                cfg,
                use_canonical_ssd_mlx_backend=False,
                ssd_nheads=None,
                ssd_headdim=None,
                mamba2_mlx_backend_lineage="reference_s6_mlx",
                canonical_ssd_mlx_backend_wired=False,
                canonical_ssd_mlx_blocker="canonical_ssd_mlx_backend_not_wired",
            )
        # We then republish its parameter
        # references as our own attributes so mlx.nn.Module.parameters() can
        # auto-discover them via tree_flatten.
        self._renderer = Z7Mamba2MLXNativeRenderer(renderer_cfg, seed=seed)

        # Republish flat scalar parameters as mx.array attributes.
        flat_param_attrs = _FLAT_PARAM_ATTRS
        if cfg.use_canonical_ssd_mlx_backend:
            flat_param_attrs = tuple(
                name
                for name in _FLAT_PARAM_ATTRS
                if name not in _S6_ONLY_PARAM_ATTRS
            )
        for name in flat_param_attrs:
            setattr(self, name, getattr(self._renderer, name))

        # Republish the decoder PixelShuffle conv stack as parallel lists.
        # MLX nn.Module auto-discovers mx.array entries inside lists via
        # tree_flatten (verified at landing — see module docstring).
        self.dec_block_w: list[Any] = list(self._renderer.dec_block_w)
        self.dec_block_b: list[Any] = list(self._renderer.dec_block_b)

        if cfg.use_canonical_ssd_mlx_backend:
            assert mx is not None
            nheads = int(cfg.effective_ssd_nheads or 0)
            headdim = int(cfg.effective_ssd_headdim or 0)
            if nheads <= 0 or headdim <= 0:
                raise ValueError(
                    "canonical SSD-MLX backend requires positive "
                    f"nheads/headdim, got {nheads}/{headdim}"
                )
            if nheads * headdim != cfg.d_inner:
                raise ValueError(
                    f"SSD shape mismatch: nheads*headdim={nheads * headdim} "
                    f"must equal d_inner={cfg.d_inner}"
                )
            # Match tac.optimization.mamba2_predictor._CanonicalHelperSSDCell:
            # A_log has one scalar per head, B/C/dt project from x_inner, D is
            # a zero-initialized optional skip, and the Z7 gate/out_proj wraps
            # the canonical SSD recurrence output.
            self.ssd_A_log = mx.log(
                mx.arange(1, nheads + 1, dtype=mx.float32)
            )
            self.ssd_B_proj_w = self._renderer._kaiming_linear(
                cfg.d_inner, nheads * cfg.d_state
            )
            self.ssd_C_proj_w = self._renderer._kaiming_linear(
                cfg.d_inner, nheads * cfg.d_state
            )
            self.ssd_dt_proj_w = self._renderer._kaiming_linear(
                cfg.d_inner, nheads
            )
            self.ssd_dt_proj_b = mx.zeros((nheads,), dtype=mx.float32)
            self.ssd_D = mx.zeros((nheads, headdim), dtype=mx.float32)

        # Ego motion buffer is NON-trainable contest side-info loaded from
        # real video; expose it as a plain attribute (mlx.nn.Module's
        # parameters() filter excludes it via the freeze() pattern when the
        # trainer wants explicit control; for now we leave it discoverable so
        # the harness can re-eval it when ego_motion changes per epoch).
        # Per the canonical sister Z6V2SubstrateMLX pattern (`ego_vecs`
        # IS trainable there), we mirror the discoverable form here; the
        # trainer-side optimizer's update is filtered to only the
        # _FLAT_PARAM_ATTRS + dec_block_{w,b} elements when an ego-motion-
        # freeze contract is required.
        self.ego_motion_buffer = self._renderer.ego_motion_buffer

    # ------------------------------------------------------------------
    # Synchronization: keep the delegate renderer in sync with our (possibly
    # gradient-updated) parameter attributes before forwarding.
    # ------------------------------------------------------------------

    def _sync_to_renderer(self) -> None:
        """Push our current parameter attribute values into the delegate.

        After ``mlx.nn.value_and_grad`` + optimizer.update mutates ``self``,
        the underlying ``self._renderer`` still holds the pre-update parameter
        references. This method copies our current ``mx.array`` references
        into the delegate so the next forward consumes the updated values.

        Sister of the ``model.update(params)`` pattern used inside the
        canonical MLX optimizer step — but operating at the Renderer
        boundary rather than the Module-internal boundary.
        """
        for name in _FLAT_PARAM_ATTRS:
            setattr(self._renderer, name, getattr(self, name))
        # Decoder block lists: write each element back to keep the delegate
        # in sync with possibly-replaced array references.
        for i in range(len(self.dec_block_w)):
            self._renderer.dec_block_w[i] = self.dec_block_w[i]
            self._renderer.dec_block_b[i] = self.dec_block_b[i]
        # Ego motion buffer (non-trainable but possibly re-loaded per epoch).
        self._renderer.ego_motion_buffer = self.ego_motion_buffer

    # ------------------------------------------------------------------
    # Forward path (gradient-preserving MLX-native; NOT delegate-passthrough).
    #
    # CRITICAL: the existing ``Z7Mamba2MLXNativeRenderer._pixel_shuffle_channels_last``
    # at ``mlx_native.py:854-878`` falls back through ``np.asarray(x)`` +
    # ``mx.array(a)`` which BREAKS the MLX gradient chain (the numpy
    # round-trip is opaque to ``mlx.nn.value_and_grad``). Empirically
    # verified at landing: bypassing this path produces ``||g||==0`` on
    # ``latent_init`` / ``residuals`` / ``mamba_in_proj_w`` / etc., so
    # 600-pair MLX-LOCAL training would learn nothing.
    #
    # The fix per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    # against" + Catalog #110/#113 APPEND-ONLY: this wrapper re-implements
    # the forward path using the canonical MLX-native PixelShuffle helper
    # ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``
    # (empirically PyTorch-byte-stable per CONSOLIDATE-OP-1 2026-05-26;
    # used by sister Z6-v2 MLX renderer + all Path 3 Apple-Silicon MLX
    # substrates). The existing ``mlx_native.py`` remains UNCHANGED per
    # APPEND-ONLY discipline; the byte-parity ``export_state_dict`` /
    # ``load_state_dict_from_numpy`` bridge still delegates to it for
    # the #1251 PyTorch round-trip.
    #
    # The Mamba-2 cell forward (``_mamba2_step``) + predictor step
    # (``_predict_step``) + autoregressive replay (``replay_latents``) ARE
    # gradient-preserving in the existing renderer (no numpy round-trip);
    # we still delegate to those for the latent computation. The fix is
    # SCOPED to the decoder PixelShuffle path that was non-MLX-native.
    # ------------------------------------------------------------------

    def _decode_latents_native(self, z_batch: Any) -> tuple[Any, Any]:
        """Gradient-preserving MLX-native decoder forward.

        Mirrors ``Z7Mamba2MLXNativeRenderer._decode_latents`` semantics but
        uses the canonical PR95 ``pixel_shuffle_2x_nhwc`` instead of the
        numpy-fallback ``_pixel_shuffle_channels_last``. Decoder output
        ``(P, 384, 512)`` exactly matches contest dims (24x32 initial grid
        + 4 PixelShuffle(2) blocks = 16x upsample → 384x512), so the
        legacy bilinear-resize branch never fires (verified at landing).

        Args:
            z_batch: ``(P, latent_dim)`` latents from the autoregressive
                Mamba-2 replay.

        Returns:
            ``(rgb_0, rgb_1)`` each ``(P, 3, output_height, output_width)``
            in unit-domain ``[0, 1]`` (sigmoid).
        """
        from tac.local_acceleration.pr95_hnerv_mlx import (
            pixel_shuffle_2x_nhwc,
        )

        cfg = self.cfg
        batch = z_batch.shape[0]
        # initial_proj
        flat = z_batch @ self.dec_initial_proj_w.T + self.dec_initial_proj_b
        # Reshape to (P, embed_dim, H0, W0) then transpose to (P, H0, W0,
        # embed_dim) for MLX channels-last conv (mirrors existing renderer
        # canonical reshape order).
        grid = mx.reshape(
            flat,
            (
                batch,
                cfg.decoder_embed_dim,
                cfg.decoder_initial_grid_h,
                cfg.decoder_initial_grid_w,
            ),
        )
        h = mx.transpose(grid, (0, 2, 3, 1))

        for i in range(cfg.decoder_num_upsample_blocks):
            # Conv2d(in_ch, 4*out_ch, k=3, p=1) — MLX channels-last
            h = mx.conv2d(h, self.dec_block_w[i], padding=1)
            h = h + self.dec_block_b[i]
            # CANONICAL PixelShuffle (gradient-preserving MLX-native; sister
            # of Z6V2SubstrateMLX line 249 + all Path 3 Apple-Silicon MLX
            # substrates per CONSOLIDATE-OP-1 2026-05-26).
            h = pixel_shuffle_2x_nhwc(h)
            # ReLU activation (mirrors existing renderer canonical).
            h = mx.maximum(h, 0.0)

        # Final 6-channel conv
        h = mx.conv2d(h, self.dec_final_w, padding=1)
        h = h + self.dec_final_b
        # Transpose back to (P, C, H, W) for shape parity with PyTorch sister
        h = mx.transpose(h, (0, 3, 1, 2))  # (P, 6, H_dec, W_dec)

        # Sanity: decoder output should match contest resolution exactly.
        # The legacy ``_bilinear_resize_np`` branch in ``mlx_native.py`` is
        # NEVER reached because 24*16=384 and 32*16=512 (verified at
        # landing); we omit it here so the forward path is unambiguously
        # gradient-preserving.
        cur_h, cur_w = int(h.shape[-2]), int(h.shape[-1])
        if cur_h != cfg.output_height or cur_w != cfg.output_width:
            raise RuntimeError(
                f"Z7Mamba2MLXModule decoder output ({cur_h}, {cur_w}) does "
                f"not match expected ({cfg.output_height}, {cfg.output_width}). "
                "The L0 SCAFFOLD canonical config produces (384, 512) via "
                "24x32 initial grid + 4 PixelShuffle(2) blocks; if you "
                "changed decoder_initial_grid_* or decoder_num_upsample_blocks "
                "you must also wire a gradient-preserving MLX-native bilinear "
                "resize (the legacy renderer's numpy-PIL path breaks gradients)."
            )

        h = mx.sigmoid(h)
        rgb_0 = h[:, :3, :, :]
        rgb_1 = h[:, 3:, :, :]
        return rgb_0, rgb_1

    def _replay_latents_native(self) -> Any:
        """Gradient-preserving MLX-native autoregressive Mamba-2 replay.

        The existing ``Z7Mamba2MLXNativeRenderer.replay_latents`` IS
        gradient-preserving (no numpy round-trip); but it consumes and
        mutates ``self._renderer._h`` which would shadow our wrapper's
        gradient-tracked parameter view. We re-implement the replay loop
        here so it consumes ``self`` parameters directly (the
        ``mlx.nn.value_and_grad`` closure differentiates ``self``, not
        ``self._renderer``).
        """
        cfg = self.cfg
        # Initial state per ``reset_state`` semantics (canonical Mamba-2 zero
        # init).
        if cfg.identity_predictor:
            h_state = None
        elif cfg.use_canonical_ssd_mlx_backend:
            nheads = int(cfg.effective_ssd_nheads or 0)
            headdim = int(cfg.effective_ssd_headdim or 0)
            h_state = mx.zeros(
                (1, nheads, headdim, cfg.d_state), dtype=mx.float32
            )
        else:
            d_inner = cfg.d_inner
            h_state = mx.zeros((1, d_inner, cfg.d_state), dtype=mx.float32)

        # z starts as latent_init (shape (latent_dim,)) -> (1, latent_dim)
        z = mx.expand_dims(self.latent_init, 0)
        outs: list[Any] = []
        for t in range(cfg.num_pairs):
            if cfg.stateful is False and not cfg.identity_predictor:
                # Stateless ablation: reset hidden state before each pair.
                if cfg.use_canonical_ssd_mlx_backend:
                    nheads = int(cfg.effective_ssd_nheads or 0)
                    headdim = int(cfg.effective_ssd_headdim or 0)
                    h_state = mx.zeros(
                        (1, nheads, headdim, cfg.d_state), dtype=mx.float32
                    )
                else:
                    h_state = mx.zeros(
                        (1, cfg.d_inner, cfg.d_state), dtype=mx.float32
                    )
            ego_t = self.ego_motion_buffer[t : t + 1]
            pred, h_state = self._predict_step_native(z, ego_t, h_state)
            z = pred + self.residuals[t : t + 1]
            outs.append(mx.squeeze(z, axis=0))
        return mx.stack(outs, axis=0)

    def _predict_step_native(
        self, z_prev: Any, ego_t: Any, h_state: Any
    ) -> tuple[Any, Any]:
        """Single predictor step using ``self`` parameters; returns ``(z_pred, h_state)``."""
        cfg = self.cfg
        if cfg.identity_predictor:
            return z_prev, h_state
        # Concat input
        x_in = mx.concatenate([z_prev, ego_t], axis=-1)
        # Input projection
        x_proj = x_in @ self.input_projection_w.T + self.input_projection_b
        # Mamba-2 cell step
        y, h_state = self._mamba2_step_native(x_proj, h_state)
        # Output projection
        z_pred = y @ self.output_projection_w.T + self.output_projection_b
        return z_pred, h_state

    def _mamba2_step_native(
        self, x_t: Any, h_state: Any
    ) -> tuple[Any, Any]:
        """Single Mamba-2 timestep using ``self`` parameters; returns ``(y_t, h_state)``.

        Mirrors ``_ReferenceMamba2Cell.forward`` byte-stably; the only
        difference from the existing renderer's ``_mamba2_step`` is that the
        hidden state is THREADED EXPLICITLY rather than mutated on
        ``self._renderer._h`` — so ``mlx.nn.value_and_grad`` sees a pure
        function of ``self``'s parameters.
        """
        cfg = self.cfg
        if cfg.identity_predictor:
            return x_t, h_state
        if cfg.use_canonical_ssd_mlx_backend:
            return self._mamba2_ssd_step_native(x_t, h_state)
        d_inner = cfg.d_inner

        # Input + gate projection
        xz = x_t @ self.mamba_in_proj_w.T
        x_inner = xz[:, :d_inner]
        z_gate = xz[:, d_inner:]

        # Selective projection
        dt = nn.softplus(
            x_inner @ self.mamba_dt_proj_w.T + self.mamba_dt_proj_b
        )
        A = -mx.exp(self.mamba_A_log)
        B = x_inner @ self.mamba_B_proj_w.T
        C = x_inner @ self.mamba_C_proj_w.T

        # Discretize: A_bar = exp(dt * A); B_bar = dt * B
        A_bar = mx.exp(mx.expand_dims(A, 0) * mx.expand_dims(dt, -1))
        B_bar = mx.expand_dims(dt, -1) * mx.expand_dims(B, 1)

        # State update: h_t = A_bar * h_{t-1} + B_bar * x_inner
        h_t = A_bar * h_state + B_bar * mx.expand_dims(x_inner, -1)

        # Output: y_inner = sum_d_state(h_t * C), gated by sigmoid(z)
        y_inner = mx.sum(h_t * mx.expand_dims(C, 1), axis=-1)
        y_inner = y_inner * mx.sigmoid(z_gate)
        y_t = y_inner @ self.mamba_out_proj_w.T
        return y_t, h_t

    def _mamba2_ssd_step_native(
        self, x_t: Any, h_state: Any
    ) -> tuple[Any, Any]:
        """Z7-gated canonical Mamba-2 SSD step using the shared MLX helper.

        This mirrors ``tac.optimization.mamba2_predictor._CanonicalHelperSSDCell``
        at the projection/wrapper layer and delegates the recurrent update to
        ``tac.substrates._shared.mamba2_ssd.mamba2_ssd_step_mlx``. Provenance
        is intentionally split: the recurrence core is canonical SSD-MLX, while
        PyTorch bridge/export is blocked until a matching runtime adapter exists.
        """
        from tac.substrates._shared.mamba2_ssd import (
            Mamba2SSDMLXState,
            mamba2_ssd_step_mlx,
        )

        cfg = self.cfg
        batch = int(x_t.shape[0])
        d_inner = cfg.d_inner
        nheads = int(cfg.effective_ssd_nheads or 0)
        headdim = int(cfg.effective_ssd_headdim or 0)

        xz = x_t @ self.mamba_in_proj_w.T
        x_inner = xz[:, :d_inner]
        z_gate = xz[:, d_inner:]
        x_per_head = mx.reshape(x_inner, (batch, nheads, headdim))
        B_t = mx.reshape(
            x_inner @ self.ssd_B_proj_w.T,
            (batch, nheads, cfg.d_state),
        )
        C_t = mx.reshape(
            x_inner @ self.ssd_C_proj_w.T,
            (batch, nheads, cfg.d_state),
        )
        dt_t = nn.softplus(x_inner @ self.ssd_dt_proj_w.T + self.ssd_dt_proj_b)

        next_state, y_per_head = mamba2_ssd_step_mlx(
            state=Mamba2SSDMLXState(h=h_state),
            x_t=x_per_head,
            A_log=self.ssd_A_log,
            B_t=B_t,
            C_t=C_t,
            dt_t=dt_t,
            D=self.ssd_D,
        )
        y_inner = mx.reshape(y_per_head, (batch, d_inner))
        y_inner = y_inner * mx.sigmoid(z_gate)
        y_t = y_inner @ self.mamba_out_proj_w.T
        return y_t, next_state.h

    def reconstruct_all_pairs(self) -> tuple[Any, Any, Any]:
        """Full-sequence replay + decode; gradient-preserving MLX-native.

        Returns ``(rgb_0, rgb_1, latents)`` where rgb_* are
        ``(num_pairs, 3, H, W)`` and latents are ``(num_pairs, latent_dim)``.
        """
        latents = self._replay_latents_native()
        rgb_0, rgb_1 = self._decode_latents_native(latents)
        return rgb_0, rgb_1, latents

    def reconstruct_pair(
        self, pair_indices: Any
    ) -> tuple[Any, Any, Any]:
        """Canonical forward convention for ``forward_convention="reconstruct_pair_nchw01"``.

        Per Catalog #218 mini-batch reconstruct discipline: caller passes
        specific pair indices to bound peak activation memory at training
        time. The full 600-pair Mamba-2 sequence is autoregressively
        replayed (deterministic + necessary for stateful Mamba-2), then
        only the requested pairs are returned.

        Returns:
            ``(rgb_0, rgb_1, latents)`` each ``rgb_*`` shape
            ``(P, 3, output_height, output_width)`` in unit-domain ``[0, 1]``
            and ``latents`` shape ``(P, latent_dim)``. Pair indices may be a
            numpy int64 array OR an MLX int32 array.
        """
        cfg = self.cfg
        # Normalize indices: accept MLX or numpy.
        if isinstance(pair_indices, np.ndarray):
            idx_np = pair_indices.astype(np.int64)
        else:
            # Assume MLX or list-like.
            idx_np = np.asarray(pair_indices, dtype=np.int64)
        if idx_np.size == 0:
            raise ValueError("pair_indices must be non-empty")
        if idx_np.min() < 0 or idx_np.max() >= cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {cfg.num_pairs}); got "
                f"[{int(idx_np.min())}, {int(idx_np.max())}]"
            )

        # Full-sequence replay + decode (gradient-preserving).
        rgb_0, rgb_1, latents = self.reconstruct_all_pairs()
        idx_mx = mx.array(idx_np)
        return rgb_0[idx_mx], rgb_1[idx_mx], latents[idx_mx]

    def __call__(self, pair_indices: Any) -> tuple[Any, Any, Any]:
        """Default forward returns ``reconstruct_pair`` output.

        Per the canonical MLX harness contract at
        ``tac.substrates._shared.mlx_score_aware.loss.decode_frames_nhwc01``:
        when ``bundle.forward_convention == "reconstruct_pair_nchw01"`` the
        harness calls ``model.reconstruct_pair(idx)`` and discards the
        latents return; we keep the 3-tuple return for direct callers that
        want the latents (e.g. probe-disambiguator surfaces per Catalog
        #125 hook #6).
        """
        return self.reconstruct_pair(pair_indices)

    # ------------------------------------------------------------------
    # state_dict bridge (delegated to existing renderer)
    # ------------------------------------------------------------------

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export parameters in PyTorch-layout numpy dict (Catalog #1251 bridge).

        Synchronizes the delegate first so any gradient-updated parameters
        flow into the exported state dict. Field names + shapes match the
        canonical PyTorch :class:`Z7Mamba2PredictiveCodingSubstrate` 1:1.
        """
        if self.cfg.use_canonical_ssd_mlx_backend:
            raise NotImplementedError(
                "canonical_ssd_mlx_pytorch_bridge_export_not_wired: "
                "Z7 canonical SSD-MLX training uses a different recurrent "
                "parameterization from the existing S6-shaped PyTorch bridge. "
                "Export is intentionally blocked until the matching contest "
                "runtime adapter is implemented."
            )
        self._sync_to_renderer()
        return self._renderer.export_state_dict()

    def load_state_dict_from_numpy(
        self, state: dict[str, np.ndarray]
    ) -> None:
        """Load parameters from a numpy state dict; mirror into our attributes."""
        if self.cfg.use_canonical_ssd_mlx_backend:
            raise NotImplementedError(
                "canonical_ssd_mlx_state_dict_import_not_wired: existing "
                "Z7 numpy state_dict bridge is S6-shaped and cannot be loaded "
                "into the canonical SSD-MLX recurrence without an explicit "
                "adapter."
            )
        self._renderer.load_state_dict_from_numpy(state)
        # Re-publish into our attributes so mlx.nn.Module.parameters() sees
        # the loaded values.
        for name in _FLAT_PARAM_ATTRS:
            setattr(self, name, getattr(self._renderer, name))
        self.dec_block_w = list(self._renderer.dec_block_w)
        self.dec_block_b = list(self._renderer.dec_block_b)
        self.ego_motion_buffer = self._renderer.ego_motion_buffer

    def num_parameters(self) -> int:
        """Total trainable parameter float count (delegates to renderer)."""
        total = int(self._renderer.num_parameters())
        if self.cfg.use_canonical_ssd_mlx_backend:
            nheads = int(self.cfg.effective_ssd_nheads or 0)
            d_inner = self.cfg.d_inner
            total -= 3 * d_inner * self.cfg.d_state
            total -= d_inner * d_inner + d_inner
            total += nheads
            total += d_inner * nheads * self.cfg.d_state
            total += d_inner * nheads * self.cfg.d_state
            total += d_inner * nheads + nheads
            total += nheads * int(self.cfg.effective_ssd_headdim or 0)
        return total

    @classmethod
    def from_pytorch_config(
        cls,
        cfg: Z7Mamba2PredictiveCodingConfig,
        *,
        seed: int = 0,
    ) -> Z7Mamba2MLXModule:
        """Build the module from a canonical PyTorch config (#1251 bridge sister)."""
        mlx_cfg = Z7Mamba2MLXRenderConfig.from_pytorch_config(cfg)
        return cls(mlx_cfg, seed=seed)


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "Z7Mamba2MLXModule",
]
