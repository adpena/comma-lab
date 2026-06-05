# SPDX-License-Identifier: MIT
"""Generic Style-B MLX adapter satisfying ``SubstrateLongTrainingAdapter``.

# NO_GRAD_WAIVED:MLX_substrate_adapter_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive

Separation of concerns: this module owns ONLY the bridge between a substrate
``RendererBundle`` and the canonical L2 harness
``tac.training.long_training_canonical.run_long_training``. It generalizes the
proven Z6 ``Z6LongTrainingAdapter`` so each substrate ``_full_main`` is ~30 LOC
of config + one harness call. The training LOOP / EMA shadow / OOM-safe step /
early-stop / telemetry / Provenance / posterior anchor all live in
``run_long_training`` (DELEGATED, not duplicated — per CLAUDE.md "Beauty,
simplicity, and developer experience" + the prompt's COMPOSE-do-not-duplicate
directive).

Style B (combined ``train_step``) is used because MLX's ``value_and_grad``
requires a combined value+grad+update step (the canonical helper prefers
``train_step`` when present per the Protocol contract).

[verified-against: tac.training.long_training_canonical.SubstrateLongTrainingAdapter Protocol]
[verified-against: tac.substrates.time_traveler_l5_z6.long_training_adapter.Z6LongTrainingAdapter proven Style-B reference]
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
    require_mlx_for_harness,
)
from tac.substrates._shared.mlx_score_aware.dual_ascent import (
    CONTEST_RATE_SCORE_PER_BYTE,
    TrainTimeDualAscentController,
    safe_dual_metric_key,
)
from tac.substrates._shared.mlx_score_aware.loss import (
    component_loss_weight,
    score_aware_loss,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS: tuple[str, ...] = (
    "adam",
    "adamw",
    "adamax",
    "adadelta",
    "adagrad",
    "aurora_like",
    "rmsprop",
    "sgd",
    "lion",
    "adafactor",
    "muon",
    "pact_muon_adamw",
)
DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND = "pact_muon_adamw"
MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS: tuple[str, ...] = (
    "adamw",
    "aurora_like",
    "sgd",
    "lion",
    "adafactor",
    "muon",
    "pact_muon_adamw",
)
_MLX_OPTIMIZER_PROVENANCE_BY_KIND: dict[str, dict[str, Any]] = {
    "adamw": {
        "borrowed_from": "mlx.optimizers.AdamW",
        "role": "legacy_default_and_pr95_adam_branch_control",
        "contest_adaptation": "score_aware_loss_qat_archive_export",
    },
    "lion": {
        "borrowed_from": "mlx.optimizers.Lion",
        "role": "native_mlx_low_state_optimizer_sweep",
        "contest_adaptation": "false_authority_neRV_score_loop_candidate",
    },
    "adafactor": {
        "borrowed_from": "mlx.optimizers.Adafactor",
        "role": "native_mlx_memory_efficient_optimizer_sweep",
        "contest_adaptation": "relative_step_disabled_for_explicit_priced_curriculum_lr",
    },
    "rmsprop": {
        "borrowed_from": "mlx.optimizers.RMSprop",
        "role": "native_mlx_curvature_baseline_sweep",
        "contest_adaptation": "weight_decay_rejected_when_requested",
    },
    "sgd": {
        "borrowed_from": "mlx.optimizers.SGD",
        "role": "native_mlx_momentum_free_baseline_sweep",
        "contest_adaptation": "score_loop_curvature_control",
    },
    "adam": {
        "borrowed_from": "mlx.optimizers.Adam",
        "role": "native_mlx_adam_without_decoupled_weight_decay_control",
        "contest_adaptation": "weight_decay_rejected_when_requested",
    },
    "adamax": {
        "borrowed_from": "mlx.optimizers.Adamax",
        "role": "native_mlx_infinity_norm_adam_family_sweep",
        "contest_adaptation": "weight_decay_rejected_when_requested",
    },
    "adagrad": {
        "borrowed_from": "mlx.optimizers.Adagrad",
        "role": "native_mlx_accumulated_gradient_baseline_sweep",
        "contest_adaptation": "weight_decay_rejected_when_requested",
    },
    "adadelta": {
        "borrowed_from": "mlx.optimizers.AdaDelta",
        "role": "native_mlx_lr_scale_adaptive_baseline_sweep",
        "contest_adaptation": "weight_decay_rejected_when_requested",
    },
    "aurora_like": {
        "borrowed_from": (
            "tilde-research/aurora-release leverage-uniform polar update "
            "ported to MLX"
        ),
        "role": "aurora_like_rectangular_matrix_optimizer_smoke",
        "contest_adaptation": (
            "matrix-like leaves receive leverage-uniform polar updates; "
            "rank-0/1 leaves use AdamW-style moments; macOS-MLX timing signal "
            "only until receiver-closed archive/runtime proof and exact replay"
        ),
    },
    "muon": {
        "borrowed_from": "mlx.optimizers.Muon",
        "role": "native_mlx_all_parameter_muon_sweep",
        "contest_adaptation": (
            "explicit non-default comparison against Pact's PR95-style "
            "partitioned Muon+AdamW path; no promotion without archive proof"
        ),
    },
    "pact_muon_adamw": {
        "borrowed_from": (
            "PR95 partition rule and tac.local_acceleration.pr95_hnerv_mlx "
            "Newton-Schulz Muon+AdamW step"
        ),
        "role": "default_pact_native_partitioned_muon_adamw_optimizer",
        "contest_adaptation": (
            "Pact-labeled MLX score-loop optimizer; Muon only for eligible "
            "hidden matrix/conv weights, AdamW for latents, heads, biases, "
            "and scalar-like params"
        ),
    },
}
PACT_MUON_ADAMW_MUON_LR_MULTIPLIER = 2.0e-4 / 3.0e-5
PACT_MUON_ADAMW_LATENT_LR_MULTIPLIER = 10.0
PR95_STAGE_BASE_SEG_WEIGHT = 100.0
PR95_STAGE_BASE_POSE_WEIGHT = 1.0
AURORA_LIKE_SOURCE_REPO = "https://github.com/tilde-research/aurora-release"
AURORA_LIKE_SOURCE_COMMIT = "7303d8cb9999d735cb12c921f3651f04bf362524"
AURORA_LIKE_PP_ITERATIONS = 2
AURORA_LIKE_PP_BETA = 0.5

DECODER_GRADIENT_SALIENCY_SCHEMA = "mlx_decoder_weight_gradient_saliency.v1"
_DECODER_SALIENCY_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latent_embed",
    "blocks",
    "feature_grids",
    "convnext_blocks",
    "head",
    "decoder",
    "injector",
)
_DECODER_SALIENCY_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latents",
    "codebook",
    "selector",
    "ema",
    "teacher",
    "student",
)


def _tree_name_to_saliency_group(raw_name: Any) -> str:
    """Normalize MLX tree paths into stable saliency group names."""

    if isinstance(raw_name, (tuple, list)):
        return ".".join(str(part) for part in raw_name if str(part))
    return str(raw_name)


def _is_decoder_weight_saliency_group(name: str) -> bool:
    """Return whether a gradient tree leaf belongs to decoder-weight saliency."""

    lowered = str(name).lower()
    if any(token in lowered for token in _DECODER_SALIENCY_EXCLUDE_SUBSTRINGS):
        return False
    return any(token in lowered for token in _DECODER_SALIENCY_INCLUDE_SUBSTRINGS)


def _nonnegative_float_or_none(value: Any) -> float | None:
    """Parse finite nonnegative telemetry values; reject malformed byte rows."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0.0:
        return None
    return out


def _active_pr95_stage_metric_weight(
    metrics: Mapping[str, Any],
    effective_weight_key: str,
    stage_weight_key: str,
) -> bool:
    """Return whether a PR95-stage metric is active for train-time controls."""

    raw = metrics.get(effective_weight_key)
    if raw is None:
        raw = metrics.get(stage_weight_key)
    if raw is None:
        return True
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _assert_mlx_loss_and_gradients_finite(
    mx: Any,
    loss_value: Any,
    grads: Any,
    *,
    context: str,
) -> dict[str, float]:
    """Fail before optimizer state is mutated by non-finite MLX gradients."""

    from mlx.utils import tree_flatten

    loss_finite = mx.all(mx.isfinite(loss_value))
    checks: list[tuple[str, Any]] = []
    for raw_name, leaf in tree_flatten(grads):
        if leaf is None:
            continue
        checks.append((_tree_name_to_saliency_group(raw_name), mx.all(mx.isfinite(leaf))))
    mx.eval(loss_finite, *(check for _name, check in checks))
    nonfinite = [name for name, check in checks if not bool(check.item())]
    if not bool(loss_finite.item()) or nonfinite:
        raise MlxScoreAwareHarnessError(
            f"{context} produced non-finite loss/gradients before optimizer "
            "update; refusing to poison optimizer state. "
            f"loss_finite={bool(loss_finite.item())}; "
            f"nonfinite_gradient_leaves={nonfinite[:16]}"
        )
    return {
        "finite_update_guard_active": 1.0,
        "finite_update_guard_checked_gradient_leaf_count": float(len(checks)),
        "finite_update_guard_nonfinite_gradient_leaf_count": 0.0,
    }


def _build_aurora_like_mlx_optimizer(
    *,
    learning_rate: Any,
    weight_decay: float | None,
) -> Any:
    """Return a real Aurora-like MLX optimizer object.

    This is a local MLX port of the public Aurora update shape: SGD momentum,
    leverage-uniform polar refinement for non-square matrices, Muon-style
    aspect-ratio scaling, and decoupled weight decay.  It is intentionally named
    ``aurora_like`` because Pact has not established source-faithful timing,
    byte-closed export, or exact replay authority for the Tilde code path.
    """

    require_mlx_for_harness()
    import mlx.core as mx
    import mlx.optimizers as mlx_optim

    class AuroraLikeMlxOptimizer(mlx_optim.Optimizer):
        def __init__(
            self,
            *,
            learning_rate: Any,
            weight_decay: float,
            mu: float = 0.95,
            nesterov: bool = True,
            pp_iterations: int = AURORA_LIKE_PP_ITERATIONS,
            pp_beta: float = AURORA_LIKE_PP_BETA,
            eps: float = 1.0e-7,
            adamw_betas: tuple[float, float] = (0.9, 0.999),
            adamw_eps: float = 1.0e-8,
        ) -> None:
            super().__init__()
            if not (0.0 < float(mu) < 1.0):
                raise ValueError(f"aurora_like mu must be in (0, 1), got {mu!r}")
            if int(pp_iterations) < 1:
                raise ValueError(
                    "aurora_like pp_iterations must be >= 1, "
                    f"got {pp_iterations!r}"
                )
            if float(pp_beta) <= 0.0:
                raise ValueError(
                    f"aurora_like pp_beta must be positive, got {pp_beta!r}"
                )
            if float(eps) <= 0.0:
                raise ValueError(f"aurora_like eps must be positive, got {eps!r}")
            self._maybe_schedule("learning_rate", learning_rate)
            self.weight_decay = float(weight_decay)
            self.mu = float(mu)
            self.nesterov = bool(nesterov)
            self.pp_iterations = int(pp_iterations)
            self.pp_beta = float(pp_beta)
            self.eps = float(eps)
            self.adamw_betas = tuple(float(value) for value in adamw_betas)
            self.adamw_eps = float(adamw_eps)
            self.source_repo = AURORA_LIKE_SOURCE_REPO
            self.source_commit = AURORA_LIKE_SOURCE_COMMIT

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            state["momentum"] = mx.zeros_like(parameter)
            state["adamw_m"] = mx.zeros_like(parameter)
            state["adamw_v"] = mx.zeros_like(parameter)

        def _polar_simple_quintic(self, matrix: Any) -> Any:
            x = matrix.astype(mx.float32)
            transposed = int(x.shape[-2]) > int(x.shape[-1])
            if transposed:
                x = x.T
            x = x / (mx.linalg.norm(x, keepdims=True) + self.eps)
            a, b, c = (2.0, -1.5, 0.5)
            for _ in range(12):
                aa = x @ x.T
                bb = b * aa + c * (aa @ aa)
                x = a * x + bb @ x
            if transposed:
                x = x.T
            return x.astype(matrix.dtype)

        def _aurora_matrix_update(self, update: Any) -> Any:
            original_shape = update.shape
            if len(original_shape) > 2:
                rows = int(original_shape[0])
                cols = math.prod(int(dim) for dim in original_shape[1:])
                update_2d = mx.reshape(update, (rows, cols))
            else:
                rows = int(original_shape[0])
                cols = int(original_shape[1])
                update_2d = update

            if rows == cols:
                projected = self._polar_simple_quintic(update_2d)
            else:
                tall = update_2d
                transposed = rows < cols
                if transposed:
                    tall = tall.T
                tall_rows = int(tall.shape[0])
                tall_cols = int(tall.shape[1])
                tall32 = tall.astype(mx.float32)
                target_row_sq = float(tall_cols) / float(tall_rows)
                row_norm = mx.sqrt(mx.sum(tall32 * tall32, axis=-1, keepdims=True))
                d = 1.0 / mx.maximum(row_norm, mx.array(self.eps, dtype=mx.float32))
                projected = tall32
                for index in range(self.pp_iterations):
                    projected = self._polar_simple_quintic(d * tall32).astype(
                        mx.float32
                    )
                    if index < self.pp_iterations - 1:
                        row_sq = mx.sum(
                            projected * projected,
                            axis=-1,
                            keepdims=True,
                        )
                        row_sq = mx.maximum(
                            row_sq,
                            mx.array(self.eps * self.eps, dtype=mx.float32),
                        )
                        d = d * ((target_row_sq / row_sq) ** self.pp_beta)
                if transposed:
                    projected = projected.T
                projected = projected.astype(update_2d.dtype)

            projected = projected * max(1.0, math.sqrt(float(rows) / float(cols)))
            if len(original_shape) > 2:
                return mx.reshape(projected, original_shape)
            return projected

        def _adamw_like_update(
            self,
            gradient: Any,
            parameter: Any,
            state: dict[str, Any],
            learning_rate_value: Any,
        ) -> Any:
            beta1, beta2 = self.adamw_betas
            m = state["adamw_m"]
            v = state["adamw_v"]
            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            state["adamw_m"] = m
            state["adamw_v"] = v
            base = parameter
            if self.weight_decay:
                base = parameter * (1.0 - learning_rate_value * self.weight_decay)
            return base - learning_rate_value * m / (mx.sqrt(v) + self.adamw_eps)

        def apply_single(
            self,
            gradient: Any,
            parameter: Any,
            state: dict[str, Any],
        ) -> Any:
            learning_rate_value = self.learning_rate.astype(gradient.dtype)
            if len(getattr(gradient, "shape", ())) < 2:
                return self._adamw_like_update(
                    gradient,
                    parameter,
                    state,
                    learning_rate_value,
                )

            momentum = state["momentum"]
            momentum = self.mu * momentum + (1.0 - self.mu) * gradient
            state["momentum"] = momentum
            update = (
                (1.0 - self.mu) * gradient + self.mu * momentum
                if self.nesterov
                else momentum
            )
            update = self._aurora_matrix_update(update)
            base = parameter
            if self.weight_decay:
                base = parameter * (1.0 - learning_rate_value * self.weight_decay)
            return base - learning_rate_value * update.astype(parameter.dtype)

    return AuroraLikeMlxOptimizer(
        learning_rate=learning_rate,
        weight_decay=0.025 if weight_decay is None else float(weight_decay),
    )


class MlxScoreAwareAdapter:
    """Generic Style-B MLX adapter satisfying ``SubstrateLongTrainingAdapter``.

    This is the substrate-AGNOSTIC bridge between any substrate ``RendererBundle``
    and the canonical L2 harness. It generalizes the proven Z6
    ``Z6LongTrainingAdapter`` so each substrate's ``_full_main`` is ~30 LOC of
    config + one harness call.
    """

    def __init__(
        self,
        bundle: RendererBundle,
        *,
        substrate_id: str,
        pr95_faithful_curriculum_enabled: bool = False,
        pr95_curriculum_total_epochs: int | None = None,
        pr95_muon_policy: str = "faithful_stage8_only",
        pr95_stage_source_weight_amplification_enabled: bool = False,
        # Wave N+11 Z7-Mamba-2 stabilizer recipe (Slot 1 RESUME 1e2b78163
        # IMPLEMENTATION-LEVEL falsification + Wave N+10 NaN-at-ep-16-18
        # signature reactivation criteria per task #1481). All defaults are
        # legacy-preserving: when no stabilizer kwarg is passed, the adapter
        # behaves identically to the pre-Wave-N+11 code path so sister
        # substrates (Z6-v2 / dreamer / sane_hnerv / etc.) are byte-stable.
        # Per CLAUDE.md "Beauty, simplicity, and developer experience" +
        # "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" Catalog #290:
        # canonical-when-it-serves; FORK-when-it-suppresses.
        # Per Gu+Dao 2023 (Mamba-2 canonical stability) + Loshchilov+Hutter
        # 2019 (Adam regularization): grad clip max_norm=1.0 + warmup linear
        # 0->lr over 5-10 epochs + weight_decay 1e-4 + EMA 0.997 is the
        # canonical smallest-perturbation stabilizer composition for
        # state-space + Adam NaN-at-specific-epoch.
        # [verified-against: Gu & Dao 2023 "Mamba: Linear-Time Sequence Modeling
        #   with Selective State Spaces" §4 Training Stability]
        # [verified-against: Loshchilov & Hutter 2019 "Decoupled Weight Decay
        #   Regularization" §4 AdamW with weight_decay default 0.01]
        # [verified-against: mlx.optimizers.clip_grad_norm + linear_schedule +
        #   cosine_decay + join_schedules canonical primitives]
        grad_clip_max_norm: float | None = None,
        warmup_epochs: int = 0,
        warmup_steps_per_epoch: int = 1,
        weight_decay: float | None = None,
        optimizer_kind: str = DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
        cosine_decay_enabled: bool = False,
        cosine_decay_total_epochs: int | None = None,
        cosine_decay_min_lr_ratio: float = 1e-2,
        train_time_dual_ascent_config: Mapping[str, Any] | None = None,
        prioritized_pair_indices: Sequence[int] | None = None,
        pair_sampling_weights: Mapping[int, float] | None = None,
        pair_sampling_default_weight: float = 1.0,
        gradient_multiplier_by_name: Mapping[str, float] | None = None,
        bias_gradient_multiplier: float | None = None,
        output_head_bias_gradient_multiplier: float = 1.0,
    ) -> None:
        """Initialize the canonical MLX score-aware adapter.

        Args:
            bundle: substrate RendererBundle (UNIQUE axis).
            substrate_id: canonical substrate id (passed to canonical posterior).
            pr95_faithful_curriculum_enabled: opt-in to PR95-faithful 8-stage
                Muon+AdamW canonical curriculum per CLAUDE.md "HNeRV /
                leaderboard-implementation parity discipline" L14 + L15 +
                the optimizer stack research memo (commit 118ddb1a4) Option A
                MINIMUM-VIABLE recommendation. Default False preserves the
                legacy single-stage path while the default optimizer is Pact's
                partitioned Muon+AdamW control.
                When True, ``train_step`` routes per-stage optimizer state
                through the canonical ``apply_pr95_mlx_optimizer_step`` via
                the canonical ``PR95FaithfulCurriculumFactory``.
            pr95_curriculum_total_epochs: total epoch budget for the PR95
                curriculum; defaults to the canonical 29,650 per L14.
                Required when ``pr95_faithful_curriculum_enabled=True``;
                ignored otherwise.
            pr95_muon_policy: Muon activation policy for the PR95 curriculum.
                ``faithful_stage8_only`` preserves PR95 source fidelity; the
                explicit contest control ``every_stage`` keeps the same PR95
                stage loss/QAT schedule but routes Muon-eligible tensors
                through the real PR95-derived Muon branch from stage 1 onward.
            pr95_stage_source_weight_amplification_enabled: opt in to PR95's
                original SegNet:PoseNet source-scale loss multiplier (100:1).
                Default False means launch controls are literal decoder-loss
                weights, so a generic HiNeRV score-aware run cannot silently
                turn ``--segnet-distillation-weight 16`` into effective 1600x
                SegNet pressure and saturate the pixel fit.
            grad_clip_max_norm: Wave N+11 stabilizer. If not None and > 0,
                applies ``mlx.optimizers.clip_grad_norm(grads, max_norm)``
                after value_and_grad but before optimizer.update. Mamba-2
                canonical value = 1.0 per Gu+Dao 2023.
            warmup_epochs: Wave N+11 stabilizer. If > 0, builds a
                ``linear_schedule(0.0, learning_rate, warmup_epochs *
                warmup_steps_per_epoch)`` lr ramp. Mamba-2 canonical = 5-10.
            warmup_steps_per_epoch: number of train_step calls per epoch
                (used only to convert warmup_epochs into warmup_steps for
                the linear_schedule). Defaults to 1 (1 step/epoch); pass
                actual batches/epoch for proper warmup at sub-epoch step
                granularity.
            weight_decay: Wave N+11 stabilizer. AdamW weight_decay override
                (None preserves AdamW default 0.01; canonical Wave N+11 =
                1e-4 per Loshchilov+Hutter 2019).
            optimizer_kind: Wave N+11 stabilizer. One of
                ``SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS``. Default
                ``pact_muon_adamw``. All supported values route to real
                ``mlx.optimizers`` classes on Apple silicon except the Pact
                default, which routes through the PR95-derived partitioned
                Muon+AdamW train-step helper. "muon" and "lion" are native MLX
                implementations of published algorithms, not Apple-specific
                algorithms; Adafactor is pinned to explicit-LR mode so stage
                curricula remain the authority.
            cosine_decay_enabled: Wave N+11 stabilizer. When True AND
                warmup_epochs > 0 AND cosine_decay_total_epochs is set,
                composes the canonical warmup + cosine-decay schedule via
                ``mlx.optimizers.join_schedules`` (matches the proven L2
                stability hardening pattern in
                ``experiments/train_substrate_z7_mamba2_v2_mlx.py``).
            cosine_decay_total_epochs: total epoch budget (used to compute
                decay_epochs = total - warmup). Required when
                ``cosine_decay_enabled=True``.
            cosine_decay_min_lr_ratio: end-of-decay lr = peak_lr * ratio
                (default 1e-2 matches the L2 hardening canonical).
            train_time_dual_ascent_config: optional projected dual-ascent
                controller over observed loss-part metrics. This is the
                train-time version of the contest rate-distortion Lagrangian:
                it can price SegNet, PoseNet, hard-pair, and coder/rate proxy
                terms with separate dual variables instead of one static scalar
                loss blend.
            prioritized_pair_indices: optional hard-pair/sensitivity pair index
                schedule consumed before random fill. This is non-authority
                training telemetry only; it lets XRay/master-gradient hard-pair
                lists steer batches without pretending sampled training covers
                the full video.
            pair_sampling_weights: optional source-pair/local-pair sampling
                weights from XRay/scorer-error telemetry. Keys are contest
                pair indices; when ``bundle.source_pair_indices`` is set they
                resolve through that source->local map. Values are finite
                non-negative sampling mass, not loss multipliers. This changes
                which real pairs train more often; it does not change scorer
                authority or create full-video replay evidence.
            pair_sampling_default_weight: baseline sampling mass for pairs not
                explicitly present in ``pair_sampling_weights``. Set to 0.0
                only for deliberate top-pair hard-focus runs.
            gradient_multiplier_by_name: optional exact parameter-name
                gradient multipliers applied after finite-gradient validation
                and before clipping/optimizer update. This is the train-time
                hook for byte/scorer waterfilling, ablation, and anti-collapse
                controls without forking the MLX/Torch/NumPy contract.
            bias_gradient_multiplier: optional multiplier for every parameter
                whose canonical name ends in ``.bias``. Exact entries in
                ``gradient_multiplier_by_name`` and the output-head convenience
                control override this value.
            output_head_bias_gradient_multiplier: convenience multiplier for
                ``head_rgb_0.bias`` and ``head_rgb_1.bias``. HiNeRV class
                collapse probes showed direct SegNet gradients can spend most
                of their update on global RGB biases while scorer argmax
                remains one-class; values < 1.0 force the optimizer to use
                spatial decoder capacity during scorer warmup.
        """
        mx = require_mlx_for_harness()
        import mlx.nn as mlx_nn
        import mlx.optimizers as mlx_optim

        self._mx = mx
        self._mlx_nn = mlx_nn
        self._mlx_optim = mlx_optim
        self.bundle = bundle
        self.model = bundle.model
        self.substrate_id = substrate_id
        self._optimizer: Any = None
        self._optimizer_lr: float | None = None
        self._last_student_head_metrics: dict[str, float] = {}

        # Wave N+11 stabilizer state (frozen at construction; train_step reads).
        if grad_clip_max_norm is not None and float(grad_clip_max_norm) <= 0.0:
            raise ValueError(
                f"grad_clip_max_norm must be None or > 0; got {grad_clip_max_norm}"
            )
        if int(warmup_epochs) < 0:
            raise ValueError(
                f"warmup_epochs must be >= 0; got {warmup_epochs}"
            )
        if int(warmup_steps_per_epoch) <= 0:
            raise ValueError(
                f"warmup_steps_per_epoch must be > 0; got {warmup_steps_per_epoch}"
            )
        optimizer_kind_text = str(optimizer_kind).lower()
        if optimizer_kind_text not in SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS:
            raise ValueError(
                "optimizer_kind must be one of "
                f"{SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS}; got {optimizer_kind!r}"
            )
        if (
            weight_decay is not None
            and optimizer_kind_text not in MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS
        ):
            raise ValueError(
                "weight_decay is only supported for optimizer_kind values "
                f"{MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS}; got "
                f"optimizer_kind={optimizer_kind!r}"
            )
        if cosine_decay_enabled:
            if int(warmup_epochs) <= 0:
                raise ValueError(
                    "cosine_decay_enabled=True requires warmup_epochs > 0"
                )
            if cosine_decay_total_epochs is None or int(cosine_decay_total_epochs) <= int(warmup_epochs):
                raise ValueError(
                    "cosine_decay_enabled=True requires "
                    "cosine_decay_total_epochs > warmup_epochs; got "
                    f"total={cosine_decay_total_epochs} warmup={warmup_epochs}"
                )
        self._wave_n11_grad_clip_max_norm: float | None = (
            float(grad_clip_max_norm) if grad_clip_max_norm is not None else None
        )
        self._wave_n11_warmup_epochs: int = int(warmup_epochs)
        self._wave_n11_warmup_steps_per_epoch: int = int(warmup_steps_per_epoch)
        self._wave_n11_weight_decay: float | None = (
            float(weight_decay) if weight_decay is not None else None
        )
        self._wave_n11_optimizer_kind: str = optimizer_kind_text
        self._wave_n11_weight_decay_supported: bool = (
            self._wave_n11_optimizer_kind
            in MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS
        )
        self._wave_n11_cosine_decay_enabled: bool = bool(cosine_decay_enabled)
        self._wave_n11_cosine_decay_total_epochs: int | None = (
            int(cosine_decay_total_epochs)
            if cosine_decay_total_epochs is not None
            else None
        )
        self._wave_n11_cosine_decay_min_lr_ratio: float = float(
            cosine_decay_min_lr_ratio
        )
        self._prioritized_pair_indices = self._normalize_prioritized_pair_indices(
            prioritized_pair_indices
        )
        self._pair_sampling_weights = self._normalize_pair_sampling_weights(
            pair_sampling_weights
        )
        self._pair_sampling_default_weight = self._normalize_pair_sampling_default_weight(
            pair_sampling_default_weight
        )
        self._gradient_multiplier_by_name = self._normalize_gradient_multipliers(
            gradient_multiplier_by_name,
            bias_gradient_multiplier=bias_gradient_multiplier,
            output_head_bias_gradient_multiplier=output_head_bias_gradient_multiplier,
        )
        self._train_time_dual_ascent = TrainTimeDualAscentController.from_config(
            train_time_dual_ascent_config
        )
        # Wave N+11 telemetry (consumed by tests + landing memo for the
        # canonical Provenance bind step).
        self._wave_n11_grad_norm_history: list[float] = []
        self._wave_n11_clipped_count: int = 0
        self._wave_n11_step_count: int = 0
        self._decoder_grad_sq_sum_by_name: dict[str, float] = {}
        self._decoder_grad_sample_count_by_name: dict[str, int] = {}
        self._decoder_grad_numel_by_name: dict[str, int] = {}
        self._decoder_grad_absmax_by_name: dict[str, float] = {}

        # PR95-faithful 8-stage Muon+AdamW canonical curriculum state per
        # CLAUDE.md L14 + L15 (Option A). Default-off preserves legacy adapter
        # behavior; opt-in routes per-stage optimizer state through canonical
        # apply_pr95_mlx_optimizer_step via PR95FaithfulCurriculumFactory.
        self._pr95_faithful_curriculum_enabled = bool(
            pr95_faithful_curriculum_enabled
        )
        self._pr95_muon_policy = str(pr95_muon_policy)
        self._pr95_stage_source_weight_amplification_enabled = bool(
            pr95_stage_source_weight_amplification_enabled
        )
        if self._pr95_muon_policy not in {"faithful_stage8_only", "every_stage"}:
            raise ValueError(
                "pr95_muon_policy must be one of "
                "('faithful_stage8_only', 'every_stage'); got "
                f"{pr95_muon_policy!r}"
            )
        self._pr95_curriculum_factory: Any = None
        self._pr95_optimizer_state: Any = None
        self._pr95_global_epoch: int = 0
        self._pr95_last_stage_verdict: Any = None
        self._pact_muon_adamw_optimizer_state: Any = None
        self._pact_muon_adamw_last_step_summary: dict[str, Any] | None = None
        if (
            self._pr95_faithful_curriculum_enabled
            or self._wave_n11_optimizer_kind == "pact_muon_adamw"
        ):
            from tac.local_acceleration.pr95_hnerv_mlx import (
                Pr95MlxOptimizerState,
            )
            if self._wave_n11_optimizer_kind == "pact_muon_adamw":
                self._pact_muon_adamw_optimizer_state = Pr95MlxOptimizerState()
        if self._pr95_faithful_curriculum_enabled:
            from tac.substrates._shared.mlx_score_aware.pr95_faithful_curriculum import (
                CANONICAL_PR95_TOTAL_EPOCHS,
                PR95FaithfulCurriculumFactory,
            )

            self._pr95_curriculum_factory = PR95FaithfulCurriculumFactory(
                total_epoch_budget=(
                    pr95_curriculum_total_epochs
                    if pr95_curriculum_total_epochs is not None
                    else CANONICAL_PR95_TOTAL_EPOCHS
                ),
                muon_policy=self._pr95_muon_policy,
            )
            self._pr95_optimizer_state = Pr95MlxOptimizerState()
        # Sibling optimizer for the learnable student head (real-scorer-bound
        # distillation path per Catalog #164). The head's ~20 params train
        # JOINTLY with the renderer: the renderer is differentiated by the
        # canonical nn.value_and_grad(self.model, ...) closure; the head's
        # weight + bias arrays are differentiated by a sibling mx.value_and_grad
        # on a {weight, bias} dict so both descend the SAME score-aware loss.
        self._head_optimizer: Any = None
        self._head_optimizer_lr: float | None = None
        self._head_opt_state: dict[str, Any] = {}
        # Sibling optimizer for the learnable POSE student head (real-PoseNet-
        # bound distillation path per Catalog #164, POSE axis). Same joint-
        # training pattern as the SegNet head: the pose head's params descend
        # the SAME pose-MSE distill loss via a sibling mx.value_and_grad step.
        self._pose_head_optimizer: Any = None
        self._pose_head_optimizer_lr: float | None = None
        self._active_loss_weights: Mapping[str, float] = {}
        self._active_curriculum_stage_name: str | None = None
        self._active_curriculum_stage_enable_qat: bool = False
        self._active_curriculum_stage_epoch: int | None = None
        self._last_train_time_section_byte_metrics: dict[str, float] = {}
        self._last_train_time_section_byte_metric_source: str | None = None

    def _post_train_step_update(self, batch: Any) -> tuple[list[Any], dict[str, float]]:
        """Run substrate-local non-gradient updates after an accepted step."""

        hook = getattr(self.model, "post_train_step_update", None)
        if not callable(hook):
            return [], {}
        result = hook(batch)
        if result is None:
            return [], {}
        if not isinstance(result, Mapping):
            return [result], {}
        raw_targets = result.get("eval_targets", [])
        if raw_targets is None:
            eval_targets: list[Any] = []
        elif isinstance(raw_targets, list):
            eval_targets = list(raw_targets)
        else:
            eval_targets = [raw_targets]
        metrics: dict[str, float] = {}
        raw_metrics = result.get("metrics", {})
        if isinstance(raw_metrics, Mapping):
            for name, value in raw_metrics.items():
                try:
                    metrics[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
        return eval_targets, metrics

    def _score_aware_loss_part_metrics(
        self,
        batch: Any,
        *,
        loss_weights: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return machine-checkable telemetry for active score-aware terms.

        The canonical loss already owns the math. This helper only exposes its
        parts after a step so long runs cannot advertise SegNet/PoseNet coupling
        while logging only substrate-local metrics such as VQ EMA updates.
        """

        mx = self._mx
        try:
            _total, parts = score_aware_loss(
                self.bundle,
                batch,
                loss_weights=loss_weights,
            )
        except Exception:
            return {"score_aware_loss_part_probe_failed": 1.0}

        out: dict[str, float] = {}
        weights = dict(self.bundle.extra_loss_weights)
        if loss_weights:
            weights.update(
                {
                    str(key): float(value)
                    for key, value in loss_weights.items()
                    if str(key)
                    not in {
                        "recon",
                        "reconstruction",
                        "distill",
                        "segnet_distill",
                        "segnet",
                        "pose_distill",
                        "posenet_distill",
                        "pose",
                        "scorer_input_guard",
                        "scorer_input_distribution_guard",
                        "value_domain_guard",
                    }
                }
            )
        recon_stage_weight = component_loss_weight(loss_weights, "recon")
        segnet_stage_weight = component_loss_weight(loss_weights, "distill")
        pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
        scorer_input_guard_stage_weight = component_loss_weight(
            loss_weights,
            "scorer_input_guard",
        )
        for name, value in parts.items():
            mx.eval(value)
            scalar = float(value.item())
            out[f"loss_part_{name}"] = scalar
            if name == "distill":
                out["loss_part_weighted_distill"] = (
                    float(self.bundle.distillation_weight)
                    * segnet_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_distill"] = segnet_stage_weight
            elif name == "segnet_direct_live_distill":
                out["loss_part_weighted_segnet_direct_live_distill"] = (
                    float(self.bundle.segnet_direct_live_distillation_weight)
                    * segnet_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_segnet_direct_live_distill"] = (
                    segnet_stage_weight
                )
            elif name == "pose_distill":
                out["loss_part_weighted_pose_distill"] = (
                    float(self.bundle.pose_distillation_weight)
                    * pose_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_pose_distill"] = pose_stage_weight
            elif name == "pose_score_term":
                weighted = (
                    float(self.bundle.pose_distillation_weight)
                    * pose_stage_weight
                    * scalar
                )
                out["loss_part_weighted_pose_score_term"] = weighted
                out["loss_part_weighted_pose_distill"] = weighted
                out["loss_part_stage_weight_pose_distill"] = pose_stage_weight
            elif name == "recon":
                out["loss_part_weighted_recon"] = recon_stage_weight * scalar
                out["loss_part_stage_weight_recon"] = recon_stage_weight
            elif name == "scorer_input_distribution_guard":
                out["loss_part_weighted_scorer_input_distribution_guard"] = (
                    float(self.bundle.scorer_input_distribution_guard_weight)
                    * scorer_input_guard_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_scorer_input_distribution_guard"] = (
                    scorer_input_guard_stage_weight
                )
                out["loss_part_config_weight_scorer_input_distribution_guard"] = (
                    float(self.bundle.scorer_input_distribution_guard_weight)
                )
            elif name == "scorer_input_contrast_floor":
                out["loss_part_weighted_scorer_input_contrast_floor"] = (
                    float(self.bundle.scorer_input_contrast_floor_weight)
                    * scorer_input_guard_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_scorer_input_contrast_floor"] = (
                    scorer_input_guard_stage_weight
                )
                out["loss_part_config_weight_scorer_input_contrast_floor"] = (
                    float(self.bundle.scorer_input_contrast_floor_weight)
                )
            elif name in weights:
                out[f"loss_part_weighted_{name}"] = float(weights[name]) * scalar
        out["score_aware_loss_parts_active"] = float(
            ("distill" in parts)
            or ("segnet_direct_live_distill" in parts)
            or ("pose_distill" in parts)
        )
        return out

    def _pr95_stage_loss_and_parts(
        self,
        *,
        batch: Any,
        stage_verdict: Any,
        model: Any,
        loss_weights: Mapping[str, float] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Build the PR95 stage-specific scorer-surrogate loss.

        This is the PR95-faithful curriculum's loss bridge: the stage verdict's
        canonical ``loss_family`` selects the SegNet margin loss, while the
        adapter supplies decoded frames, hard SegNet targets, PoseNet targets,
        and optional C1a entropy pressure. The result is still macOS-MLX
        research signal only; archive/runtime replay remains the authority.
        """

        mx = self._mx
        from tac.local_acceleration.pr95_hnerv_mlx_stage_losses import (
            PR95_MLX_STAGE_SCORER_LOSS_SURFACE,
            PR95_SEG_LOSS_CE,
            PR95_SEG_LOSS_L7_SOFTPLUS,
            PR95_SEG_LOSS_SMOOTH_DISAGREEMENT,
            PR95_SEG_LOSS_TAU_SOFTPLUS,
            pr95_mlx_stage_seg_loss,
        )
        from tac.substrates._shared.mlx_score_aware.loss import (
            _apply_eval_roundtrip_ste_nhwc01,
            _direct_live_segnet_logit_distillation_loss_and_metrics,
            _pose_distillation_loss_and_raw_mse,
            _prepare_recon_pixel_weight,
            _weighted_recon,
            decode_frames_nhwc01,
            pose_student_inputs_nhwc,
            scorer_input_contrast_floor_loss,
            scorer_input_distribution_guard_loss,
        )

        (
            seg_control_multiplier,
            pose_control_multiplier,
            effective_seg_weight,
            effective_pose_weight,
        ) = self._pr95_stage_score_weight_controls(loss_weights=loss_weights)
        seg_stage_weight = component_loss_weight(loss_weights, "distill")
        pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
        direct_live_weight = float(
            self.bundle.segnet_direct_live_distillation_weight
        )
        direct_live_active = bool(
            direct_live_weight > 0.0 and seg_stage_weight != 0.0
        )
        seg_surrogate_active = bool(float(effective_seg_weight) != 0.0)
        pose_surrogate_active = bool(float(effective_pose_weight) != 0.0)

        if self.bundle.scorer_teacher is None or (
            seg_surrogate_active and self.bundle.learnable_student_head is None
        ):
            raise ValueError(
                "PR95-faithful stage loss requires a real SegNet teacher and "
                "learnable_student_head; refusing to fall back to generic "
                "score_aware_loss."
            )
        if (
            pose_surrogate_active
            and (
                self.bundle.pose_scorer_teacher is None
                or self.bundle.learnable_pose_student_head is None
            )
        ):
            raise ValueError(
                "PR95-faithful stage loss requires a real PoseNet teacher and "
                "learnable_pose_student_head; refusing false-authority "
                "SegNet-only stage scoring."
            )

        rgb_0, rgb_1 = decode_frames_nhwc01(self.bundle, batch)
        rgb_0 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_0)
        rgb_1 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_1)
        gt_0 = self.bundle.target_rgb_0[batch]
        gt_1 = self.bundle.target_rgb_1[batch]
        if self.bundle.recon_pixel_weight is None:
            recon = mx.mean((rgb_0 - gt_0) ** 2) + mx.mean((rgb_1 - gt_1) ** 2)
        else:
            weight_0 = mx.stop_gradient(
                _prepare_recon_pixel_weight(
                    self.bundle,
                    rgb_0.shape,
                    idx=batch,
                    frame_index=0,
                )
            )
            weight_1 = mx.stop_gradient(
                _prepare_recon_pixel_weight(
                    self.bundle,
                    rgb_1.shape,
                    idx=batch,
                    frame_index=1,
                )
            )
            recon = (
                _weighted_recon(self.bundle, rgb_0, gt_0, weight_0)
                + _weighted_recon(self.bundle, rgb_1, gt_1, weight_1)
            )

        seg_rgb = rgb_1 if self.bundle.segnet_teacher_frame_index == 1 else rgb_0
        if seg_surrogate_active:
            student_logits_nhwc = self.bundle.learnable_student_head(seg_rgb)
            teacher_logits_nhwc = mx.stop_gradient(
                self.bundle.scorer_teacher.teacher_logits_for_indices(batch)
            )
            if tuple(student_logits_nhwc.shape) != tuple(teacher_logits_nhwc.shape):
                raise ValueError(
                    "PR95-faithful SegNet student/teacher shape mismatch: "
                    f"student={tuple(student_logits_nhwc.shape)} "
                    f"teacher={tuple(teacher_logits_nhwc.shape)}"
                )
            targets_hard_nhw = mx.argmax(teacher_logits_nhwc, axis=-1)
            seg_logits_nchw = mx.transpose(student_logits_nhwc, (0, 3, 1, 2))
        else:
            seg_logits_nchw = None
            targets_hard_nhw = None

        if pose_surrogate_active:
            pose_rgb_0, pose_rgb_1 = pose_student_inputs_nhwc(
                self.bundle, rgb_0, rgb_1
            )
            pose_pred = self.bundle.learnable_pose_student_head(
                pose_rgb_0, pose_rgb_1
            )
            pose_target = mx.stop_gradient(
                self.bundle.pose_scorer_teacher.teacher_pose_for_indices(batch)
            )
            if int(pose_pred.shape[-1]) < 6 or int(pose_target.shape[-1]) < 6:
                raise ValueError(
                    "PR95-faithful PoseNet stage loss requires at least 6 pose "
                    f"dims; got pred={tuple(pose_pred.shape)} "
                    f"target={tuple(pose_target.shape)}"
                )
        else:
            pose_pred = None
            pose_target = None

        extra_qat_total, extra_qat_parts = self._extra_loss_terms_and_weighted_total(
            model,
            batch,
            qat_active=bool(stage_verdict.qat_active),
        )
        cat_entropy_term = None
        if float(stage_verdict.cat_lambda) > 0.0:
            if "coder_qat_c1a_entropy" in extra_qat_parts:
                cat_entropy_term = extra_qat_parts["coder_qat_c1a_entropy"]
            else:
                from tac.substrates._shared.mlx_score_aware.coder_qat import (
                    CoderAwareQATConfig,
                    build_decoder_c1a_entropy_term,
                )

                cat_entropy_term = build_decoder_c1a_entropy_term(
                    model,
                    CoderAwareQATConfig(enabled=True, quant_bits=8),
                    sigma=float(stage_verdict.cat_sigma),
                )

        loss_family = str(stage_verdict.loss_family)
        if seg_surrogate_active:
            seg_loss = pr95_mlx_stage_seg_loss(
                loss_family,
                seg_logits_nchw,
                targets_hard_nhw,
            )
        else:
            seg_loss = mx.array(0.0, dtype=mx.float32)
        direct_live_seg_loss = None
        direct_live_metrics: dict[str, Any] = {}
        if direct_live_active:
            (
                direct_live_seg_loss,
                direct_live_metrics,
            ) = _direct_live_segnet_logit_distillation_loss_and_metrics(
                self.bundle,
                seg_rgb,
                batch,
            )
        if pose_surrogate_active:
            per_dim_scale = getattr(
                self.bundle.pose_scorer_teacher,
                "per_dim_scale",
                None,
            )
            pose_distill, pose_distill_raw_mse = _pose_distillation_loss_and_raw_mse(
                self.bundle,
                student_pose=pose_pred[:, :6],
                teacher_pose=pose_target[:, :6],
                per_dim_scale=per_dim_scale,
            )
            pose_loss = mx.sqrt(10.0 * pose_distill + 1.0e-12)
        else:
            pose_loss = mx.array(0.0, dtype=mx.float32)
            pose_distill_raw_mse = mx.array(0.0, dtype=mx.float32)
        recon_stage_weight = component_loss_weight(loss_weights, "recon")
        scorer_input_guard_stage_weight = component_loss_weight(
            loss_weights,
            "scorer_input_guard",
        )
        total = (
            float(effective_seg_weight) * seg_loss
            + float(effective_pose_weight) * pose_loss
            + recon_stage_weight * recon
        )
        if direct_live_seg_loss is not None:
            total = (
                total
                + direct_live_weight * seg_stage_weight * direct_live_seg_loss
            )
        guard_parts: dict[str, Any] = {}
        if (
            self.bundle.scorer_input_distribution_guard_weight > 0.0
            and scorer_input_guard_stage_weight != 0.0
        ):
            guard, guard_parts = scorer_input_distribution_guard_loss(
                self.bundle,
                rgb_0,
                rgb_1,
                gt_0,
                gt_1,
            )
            total = (
                total
                + float(self.bundle.scorer_input_distribution_guard_weight)
                * scorer_input_guard_stage_weight
                * guard
            )
        contrast_floor_parts: dict[str, Any] = {}
        if (
            self.bundle.scorer_input_contrast_floor_weight > 0.0
            and scorer_input_guard_stage_weight != 0.0
        ):
            contrast_floor, contrast_floor_parts = scorer_input_contrast_floor_loss(
                self.bundle,
                rgb_0,
                rgb_1,
                gt_0,
                gt_1,
            )
            total = (
                total
                + float(self.bundle.scorer_input_contrast_floor_weight)
                * scorer_input_guard_stage_weight
                * contrast_floor
            )
        if cat_entropy_term is not None and float(stage_verdict.cat_lambda) > 0.0:
            total = total + float(stage_verdict.cat_lambda) * cat_entropy_term
        if extra_qat_total is not None:
            total = total + extra_qat_total
        family_index = {
            PR95_SEG_LOSS_CE: 1.0,
            PR95_SEG_LOSS_TAU_SOFTPLUS: 2.0,
            PR95_SEG_LOSS_SMOOTH_DISAGREEMENT: 3.0,
            PR95_SEG_LOSS_L7_SOFTPLUS: 4.0,
        }.get(loss_family, -1.0)
        parts: dict[str, Any] = {
            "pr95_stage_scorer_surrogate": total,
            "pr95_stage_seg_surrogate": seg_loss,
            "pr95_stage_pose_surrogate": pose_loss,
            "pr95_stage_pose_raw_mse": pose_distill_raw_mse,
            "pr95_stage_recon": recon,
            "pr95_stage_loss_family_index": mx.array(
                family_index, dtype=mx.float32
            ),
            "pr95_stage_loss_surface_active": mx.array(1.0, dtype=mx.float32),
            "pr95_stage_seg_control_multiplier": mx.array(
                seg_control_multiplier, dtype=mx.float32
            ),
            "pr95_stage_pose_control_multiplier": mx.array(
                pose_control_multiplier, dtype=mx.float32
            ),
            "pr95_stage_distill_weight": mx.array(
                seg_stage_weight, dtype=mx.float32
            ),
            "pr95_stage_pose_distill_weight": mx.array(
                pose_stage_weight, dtype=mx.float32
            ),
            "pr95_stage_effective_seg_weight": mx.array(
                effective_seg_weight, dtype=mx.float32
            ),
            "pr95_stage_effective_pose_weight": mx.array(
                effective_pose_weight, dtype=mx.float32
            ),
            "pr95_stage_recon_weight": mx.array(
                recon_stage_weight, dtype=mx.float32
            ),
            "pr95_stage_scorer_input_guard_weight": mx.array(
                scorer_input_guard_stage_weight, dtype=mx.float32
            ),
        }
        for name, value in guard_parts.items():
            parts[f"pr95_stage_{name}"] = value
        for name, value in contrast_floor_parts.items():
            parts[f"pr95_stage_{name}"] = value
        if cat_entropy_term is not None:
            parts["pr95_c1a_entropy"] = cat_entropy_term
        if direct_live_seg_loss is not None:
            parts["pr95_stage_segnet_direct_live_distill"] = direct_live_seg_loss
            for name, value in direct_live_metrics.items():
                parts[f"pr95_stage_{name}"] = value
        parts.update(extra_qat_parts)
        # Keep the symbolic surface reachable for tests and downstream source
        # audits without putting a string into float-only metrics.
        self._pr95_last_stage_loss_surface = PR95_MLX_STAGE_SCORER_LOSS_SURFACE
        return total, parts

    def _pr95_stage_score_weight_controls(
        self,
        *,
        loss_weights: Mapping[str, float] | None = None,
    ) -> tuple[float, float, float, float]:
        """Return launch-controlled PR95 scorer weights for decoder training.

        Generic score-aware launches use literal operator controls as decoder
        loss weights. PR95's source-faithful scorer Lagrangian (SegNet:PoseNet
        ``100:1``) is available only when explicitly opted in; otherwise it can
        dominate reconstruction by orders of magnitude and produce byte-closed
        but saturated HiNeRV packets.
        """

        seg_control = float(getattr(self.bundle, "distillation_weight", 0.0))
        pose_control = float(
            getattr(self.bundle, "pose_distillation_weight", 0.0)
        )
        seg_base = (
            PR95_STAGE_BASE_SEG_WEIGHT
            if self._pr95_stage_source_weight_amplification_enabled
            else 1.0
        )
        pose_base = (
            PR95_STAGE_BASE_POSE_WEIGHT
            if self._pr95_stage_source_weight_amplification_enabled
            else 1.0
        )
        seg_stage_weight = component_loss_weight(loss_weights, "distill")
        pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
        return (
            seg_control,
            pose_control,
            seg_base * seg_control * seg_stage_weight,
            pose_base * pose_control * pose_stage_weight,
        )

    def _extra_loss_terms_and_weighted_total(
        self,
        model: Any,
        batch: Any,
        *,
        qat_active: bool = True,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Return configured bundle extra losses and their weighted sum.

        PR95 faithful training must not advertise C1a/QAT controls that only
        exist in metadata. The shared RendererBundle already carries the real
        differentiable coder-pressure callback and weights, so the PR95 path
        consumes that same surface instead of duplicating a PR95-only shim.
        """

        if not bool(qat_active) or self.bundle.extra_loss_terms is None:
            return None, {}
        mx = self._mx
        extra = dict(self.bundle.extra_loss_terms(model, batch))
        weights = dict(self.bundle.extra_loss_weights)
        total = None
        parts: dict[str, Any] = {}
        for name, value in extra.items():
            key = str(name)
            parts[key] = value
            weight = float(weights.get(key, 0.0))
            if weight == 0.0:
                continue
            term = value * weight
            total = term if total is None else total + term
        if total is None and parts:
            total = mx.array(0.0, dtype=mx.float32)
        return total, parts

    def _pr95_stage_loss_part_metrics(
        self,
        batch: Any,
        *,
        stage_verdict: Any,
        loss_weights: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return float telemetry for the active PR95 stage loss parts."""

        mx = self._mx
        try:
            _total, parts = self._pr95_stage_loss_and_parts(
                batch=batch,
                stage_verdict=stage_verdict,
                model=self.model,
                loss_weights=loss_weights,
            )
        except Exception:
            return {"pr95_stage_loss_part_probe_failed": 1.0}

        out: dict[str, float] = {}
        (
            seg_control_multiplier,
            pose_control_multiplier,
            effective_seg_weight,
            effective_pose_weight,
        ) = self._pr95_stage_score_weight_controls(loss_weights=loss_weights)
        for name, value in parts.items():
            mx.eval(value)
            scalar = float(value.item())
            out[f"loss_part_{name}"] = scalar
            if name == "pr95_stage_seg_surrogate":
                out["loss_part_weighted_pr95_stage_seg_surrogate"] = (
                    effective_seg_weight * scalar
                )
            elif name == "pr95_stage_pose_surrogate":
                out["loss_part_weighted_pr95_stage_pose_surrogate"] = (
                    effective_pose_weight * scalar
                )
            elif name == "pr95_stage_segnet_direct_live_distill":
                direct_live_stage_weight = component_loss_weight(
                    loss_weights,
                    "distill",
                )
                direct_live_config_weight = float(
                    self.bundle.segnet_direct_live_distillation_weight
                )
                weighted = (
                    direct_live_config_weight
                    * direct_live_stage_weight
                    * scalar
                )
                out[
                    "loss_part_weighted_pr95_stage_segnet_direct_live_distill"
                ] = weighted
                out[
                    "loss_part_stage_weight_pr95_stage_segnet_direct_live_distill"
                ] = direct_live_stage_weight
                out[
                    "loss_part_config_weight_pr95_stage_segnet_direct_live_distill"
                ] = direct_live_config_weight
                if direct_live_config_weight > 0.0 and direct_live_stage_weight > 0.0:
                    out["loss_part_segnet_direct_live_distill"] = scalar
                    out["loss_part_weighted_segnet_direct_live_distill"] = weighted
                    out["loss_part_stage_weight_segnet_direct_live_distill"] = (
                        direct_live_stage_weight
                    )
                    out["loss_part_config_weight_segnet_direct_live_distill"] = (
                        direct_live_config_weight
                    )
            elif name.startswith("pr95_stage_segnet_direct_live_"):
                generic_name = name.replace(
                    "pr95_stage_segnet_direct_live_",
                    "segnet_direct_live_",
                    1,
                )
                out[f"loss_part_{generic_name}"] = scalar
            elif name == "pr95_c1a_entropy":
                out["loss_part_weighted_pr95_c1a_entropy"] = (
                    float(stage_verdict.cat_lambda) * scalar
                )
            elif name == "pr95_stage_scorer_input_distribution_guard":
                guard_stage_weight = component_loss_weight(
                    loss_weights,
                    "scorer_input_guard",
                )
                out[
                    "loss_part_weighted_pr95_stage_scorer_input_distribution_guard"
                ] = (
                    float(self.bundle.scorer_input_distribution_guard_weight)
                    * guard_stage_weight
                    * scalar
                )
                out[
                    "loss_part_stage_weight_pr95_stage_scorer_input_distribution_guard"
                ] = guard_stage_weight
                out[
                    "loss_part_config_weight_pr95_stage_scorer_input_distribution_guard"
                ] = float(self.bundle.scorer_input_distribution_guard_weight)
            elif name == "pr95_stage_scorer_input_contrast_floor":
                guard_stage_weight = component_loss_weight(
                    loss_weights,
                    "scorer_input_guard",
                )
                weighted = (
                    float(self.bundle.scorer_input_contrast_floor_weight)
                    * guard_stage_weight
                    * scalar
                )
                out[
                    "loss_part_weighted_pr95_stage_scorer_input_contrast_floor"
                ] = weighted
                out[
                    "loss_part_stage_weight_pr95_stage_scorer_input_contrast_floor"
                ] = guard_stage_weight
                out[
                    "loss_part_config_weight_pr95_stage_scorer_input_contrast_floor"
                ] = float(self.bundle.scorer_input_contrast_floor_weight)
                if (
                    float(self.bundle.scorer_input_contrast_floor_weight) > 0.0
                    and guard_stage_weight > 0.0
                ):
                    out["loss_part_scorer_input_contrast_floor"] = scalar
                    out["loss_part_weighted_scorer_input_contrast_floor"] = weighted
                    out[
                        "loss_part_stage_weight_scorer_input_contrast_floor"
                    ] = guard_stage_weight
                    out[
                        "loss_part_config_weight_scorer_input_contrast_floor"
                    ] = float(self.bundle.scorer_input_contrast_floor_weight)
            elif name.startswith("pr95_stage_scorer_input_contrast_floor_"):
                generic_name = name.replace(
                    "pr95_stage_scorer_input_contrast_floor_",
                    "scorer_input_contrast_floor_",
                    1,
                )
                out[f"loss_part_{generic_name}"] = scalar
            elif name in self.bundle.extra_loss_weights:
                out[f"loss_part_weighted_{name}"] = (
                    float(self.bundle.extra_loss_weights[name]) * scalar
                )
        out["pr95_stage_loss_parts_active"] = 1.0
        out["pr95_stage_seg_control_multiplier"] = seg_control_multiplier
        out["pr95_stage_pose_control_multiplier"] = pose_control_multiplier
        out["pr95_stage_effective_seg_weight"] = effective_seg_weight
        out["pr95_stage_effective_pose_weight"] = effective_pose_weight
        return out

    def _train_student_heads(
        self,
        *,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> list[Any]:
        """Train real-scorer student heads under the active stage weights."""

        mx = self._mx
        mlx_optim = self._mlx_optim
        eval_targets: list[Any] = []
        student_metrics: dict[str, float] = {
            "segnet_student_head_update_active": 0.0,
            "segnet_student_live_calibration_active": 0.0,
            "segnet_student_live_calibration_teacher_available": 0.0,
            "segnet_student_live_calibration_weight": float(
                self.bundle.segnet_student_live_calibration_weight
            ),
        }

        segnet_stage_weight = component_loss_weight(loss_weights, "distill")
        head = self.bundle.learnable_student_head
        if (
            self.bundle.distillation_weight > 0.0
            and segnet_stage_weight != 0.0
            and self.bundle.scorer_teacher is not None
            and head is not None
        ):
            if (
                self._head_optimizer is None
                or self._head_optimizer_lr != learning_rate
            ):
                self._head_optimizer = mlx_optim.AdamW(learning_rate=learning_rate)
                self._head_optimizer_lr = learning_rate
                self._head_opt_state = {}

            live_calibration_weight = float(
                self.bundle.segnet_student_live_calibration_weight
            )
            live_teacher_fn = None
            if live_calibration_weight > 0.0:
                live_teacher_fn = getattr(
                    self.bundle.scorer_teacher,
                    "teacher_logits_for_frames_nhwc01",
                    None,
                )
                if not callable(live_teacher_fn):
                    raise ValueError(
                        "segnet_student_live_calibration_weight > 0 but "
                        "scorer_teacher has no teacher_logits_for_frames_nhwc01 "
                        "candidate-frame teacher surface."
                    )
                student_metrics[
                    "segnet_student_live_calibration_teacher_available"
                ] = 1.0

            last_target_distill: Any | None = None
            last_live_distill: Any | None = None

            def _head_loss_fn(head_params: Mapping[str, Any]) -> Any:
                nonlocal last_live_distill, last_target_distill
                from tac.substrates._shared.mlx_score_aware.loss import (
                    _apply_eval_roundtrip_ste_nhwc01,
                    decode_frames_nhwc01,
                )
                from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                    HintonMlxCustomLossFnConfig,
                    score_teacher_distillation_loss,
                )

                rgb_0, rgb_1 = decode_frames_nhwc01(self.bundle, batch)
                rgb_0 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_0)
                rgb_1 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_1)
                seg_rgb = (
                    rgb_1
                    if self.bundle.segnet_teacher_frame_index == 1
                    else rgb_0
                )
                seg_rgb = mx.stop_gradient(seg_rgb)
                student = (
                    mx.einsum("bhwc,ck->bhwk", seg_rgb, head_params["weight"])
                    + head_params["bias"]
                )
                teacher = mx.stop_gradient(
                    self.bundle.scorer_teacher.teacher_logits_for_indices(batch)
                )
                loss_cfg = HintonMlxCustomLossFnConfig(
                    temperature=self.bundle.distillation_temperature,
                    distillation_objective=self.bundle.segnet_distillation_objective,
                    tau_boundary=self.bundle.segnet_tau_boundary,
                    hinge_margin=self.bundle.segnet_hinge_margin,
                    student_head_out_channels=self.bundle.distillation_num_classes,
                )
                distill = score_teacher_distillation_loss(
                    student_logits=student,
                    teacher_logits=teacher,
                    config=loss_cfg,
                )
                live_distill = mx.array(0.0, dtype=mx.float32)
                if live_teacher_fn is not None:
                    live_teacher = mx.stop_gradient(live_teacher_fn(seg_rgb))
                    if tuple(student.shape) != tuple(live_teacher.shape):
                        raise ValueError(
                            "SegNet live-calibration student/teacher shape "
                            "mismatch: "
                            f"student={tuple(student.shape)} "
                            f"teacher={tuple(live_teacher.shape)}"
                        )
                    live_distill = score_teacher_distillation_loss(
                        student_logits=student,
                        teacher_logits=live_teacher,
                        config=loss_cfg,
                    )
                last_target_distill = distill
                last_live_distill = live_distill
                return (
                    float(self.bundle.distillation_weight)
                    * segnet_stage_weight
                    * (distill + live_calibration_weight * live_distill)
                )

            head_params = {"weight": head.weight, "bias": head.bias}
            _hloss, hgrads = mx.value_and_grad(_head_loss_fn)(head_params)
            self._head_optimizer.update(head_params, hgrads)
            head.weight = head_params["weight"]
            head.bias = head_params["bias"]
            metric_targets = [
                _hloss,
                last_target_distill,
                last_live_distill,
                head.weight,
                head.bias,
            ]
            mx.eval(*(target for target in metric_targets if target is not None))
            student_metrics["segnet_student_head_update_active"] = 1.0
            student_metrics["loss_part_segnet_student_head_total"] = float(
                _hloss.item()
            )
            if last_target_distill is not None:
                target_scalar = float(last_target_distill.item())
                student_metrics["loss_part_segnet_student_head_target_distill"] = (
                    target_scalar
                )
                student_metrics[
                    "loss_part_weighted_segnet_student_head_target_distill"
                ] = (
                    float(self.bundle.distillation_weight)
                    * segnet_stage_weight
                    * target_scalar
                )
            if last_live_distill is not None and live_calibration_weight > 0.0:
                live_scalar = float(last_live_distill.item())
                student_metrics["segnet_student_live_calibration_active"] = 1.0
                student_metrics["loss_part_segnet_student_live_calibration"] = (
                    live_scalar
                )
                student_metrics[
                    "loss_part_weighted_segnet_student_live_calibration"
                ] = (
                    float(self.bundle.distillation_weight)
                    * segnet_stage_weight
                    * live_calibration_weight
                    * live_scalar
                )
            eval_targets.extend(
                [head.weight, head.bias, self._head_optimizer.state]
            )

        pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
        pose_head = self.bundle.learnable_pose_student_head
        if (
            self.bundle.pose_distillation_weight > 0.0
            and pose_stage_weight != 0.0
            and self.bundle.pose_scorer_teacher is not None
            and pose_head is not None
        ):
            if (
                self._pose_head_optimizer is None
                or self._pose_head_optimizer_lr != learning_rate
            ):
                self._pose_head_optimizer = mlx_optim.AdamW(
                    learning_rate=learning_rate
                )
                self._pose_head_optimizer_lr = learning_rate

            def _pose_head_loss_fn(pose_params: Mapping[str, Any]) -> Any:
                from tac.substrates._shared.mlx_score_aware.loss import (
                    _apply_eval_roundtrip_ste_nhwc01,
                    _pose_distillation_loss_and_raw_mse,
                    decode_frames_nhwc01,
                    pose_student_inputs_nhwc,
                )

                rgb_0, rgb_1 = decode_frames_nhwc01(self.bundle, batch)
                rgb_0 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_0)
                rgb_1 = _apply_eval_roundtrip_ste_nhwc01(self.bundle, rgb_1)
                pose_rgb_0, pose_rgb_1 = pose_student_inputs_nhwc(
                    self.bundle, rgb_0, rgb_1
                )
                student_pose = pose_head.forward_with_params(
                    mx.stop_gradient(pose_rgb_0),
                    mx.stop_gradient(pose_rgb_1),
                    {
                        "weight": pose_params["weight"],
                        "bias": pose_params["bias"],
                    },
                )
                teacher_pose = mx.stop_gradient(
                    self.bundle.pose_scorer_teacher.teacher_pose_for_indices(batch)
                )
                pose_distill, _raw_mse = _pose_distillation_loss_and_raw_mse(
                    self.bundle,
                    student_pose=student_pose,
                    teacher_pose=teacher_pose,
                    per_dim_scale=getattr(
                        self.bundle.pose_scorer_teacher,
                        "per_dim_scale",
                        None,
                    ),
                )
                pose_score_term = mx.sqrt(10.0 * pose_distill + 1.0e-12)
                return (
                    float(self.bundle.pose_distillation_weight)
                    * pose_stage_weight
                    * pose_score_term
                )

            pose_params = {"weight": pose_head.weight, "bias": pose_head.bias}
            _ploss, pgrads = mx.value_and_grad(_pose_head_loss_fn)(pose_params)
            self._pose_head_optimizer.update(pose_params, pgrads)
            pose_head.weight = pose_params["weight"]
            pose_head.bias = pose_params["bias"]
            eval_targets.extend(
                [pose_head.weight, pose_head.bias, self._pose_head_optimizer.state]
            )

        self._last_student_head_metrics = student_metrics
        return eval_targets

    def artifact_metadata(self) -> Mapping[str, Any]:
        """Return non-authority substrate metadata for TrainingArtifact JSON.

        This is the canonical MLX harness bridge for substrate-local facts
        such as backend lineage or math-fidelity blockers. Readiness and score
        authority remain owned by ``TrainingArtifact`` itself; the bundle
        refuses those duplicate keys at construction.
        """
        metadata = dict(self.bundle.substrate_artifact_metadata)
        if "score_aware_training" in metadata:
            metadata["substrate_supplied_score_aware_training"] = metadata.pop(
                "score_aware_training"
            )
        metadata["score_aware_training"] = {
            "schema": "mlx_score_aware_training_objective.v1",
            "segnet_distillation_objective": self.bundle.segnet_distillation_objective,
            "segnet_tau_boundary": float(self.bundle.segnet_tau_boundary),
            "segnet_hinge_margin": float(self.bundle.segnet_hinge_margin),
            "segnet_student_live_calibration": {
                "schema": "mlx_score_aware_segnet_student_live_calibration.v1",
                "enabled": (
                    float(self.bundle.segnet_student_live_calibration_weight) > 0.0
                    and self.bundle.scorer_teacher is not None
                    and self.bundle.distillation_weight > 0.0
                ),
                "weight": float(
                    self.bundle.segnet_student_live_calibration_weight
                ),
                "teacher_surface": "teacher_logits_for_frames_nhwc01",
                "candidate_frame_domain": "decoded_eval_roundtrip_nhwc01",
                "purpose": (
                    "calibrate_student_head_to_real_segnet_candidate_response_"
                    "before_renderer_uses_student_surrogate_gradients"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "segnet_direct_live_distillation": {
                "schema": "mlx_score_aware_segnet_direct_live_distillation.v1",
                "enabled": (
                    float(self.bundle.segnet_direct_live_distillation_weight)
                    > 0.0
                    and self.bundle.scorer_teacher is not None
                ),
                "weight": float(
                    self.bundle.segnet_direct_live_distillation_weight
                ),
                "base_loss_weight": float(
                    self.bundle.segnet_direct_live_base_loss_weight
                ),
                "class_histogram_weight": float(
                    self.bundle.segnet_direct_live_class_histogram_weight
                ),
                "class_balanced_hinge_weight": float(
                    self.bundle.segnet_direct_live_class_balanced_hinge_weight
                ),
                "class_balanced_ce_weight": float(
                    self.bundle.segnet_direct_live_class_balanced_ce_weight
                ),
                "teacher_surface": "teacher_logits_for_frames_nhwc01",
                "candidate_frame_domain": "decoded_eval_roundtrip_nhwc01",
                "objective": str(self.bundle.segnet_distillation_objective),
                "loss": (
                    "real_segnet_live_logits_default_mse_or_configured_"
                    "argmax_hinge_plus_optional_target_class_histogram_or_"
                    "class_balanced_bootstrap_tether_or_class_balanced_ce"
                ),
                "purpose": (
                    "backpropagate_real_segnet_input_vjp_into_renderer_pixels_"
                    "when_student_surrogate_collapses_to_one_class"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "scorer_input_contrast_floor": {
                "schema": "mlx_score_aware_scorer_input_contrast_floor.v1",
                "enabled": (
                    float(self.bundle.scorer_input_contrast_floor_weight) > 0.0
                ),
                "weight": float(self.bundle.scorer_input_contrast_floor_weight),
                "segnet_last_rgb_min_std_ratio": float(
                    self.bundle.scorer_input_contrast_floor_segnet_min_std_ratio
                ),
                "posenet_yuv6_pair_min_std_ratio": float(
                    self.bundle.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
                ),
                "domains": {
                    "segnet": "last_frame_rgb_nhwc01",
                    "posenet": "two_frame_pr95_yuv6_pair_nhwc01",
                },
                "purpose": (
                    "refuse_flat_scorer_input_basin_before_live_segnet_or_pose_"
                    "surrogates_optimize_inside_argmax_collapse"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "segnet_teacher_frame_index": int(self.bundle.segnet_teacher_frame_index),
            "segnet_distillation_weight": float(self.bundle.distillation_weight),
            "pose_distillation_weight": float(self.bundle.pose_distillation_weight),
            "pose_distillation_loss": str(self.bundle.pose_distillation_loss),
            "pose_distillation_huber_delta": float(
                self.bundle.pose_distillation_huber_delta
            ),
            "pose_dims": int(self.bundle.pose_dims),
            "has_real_segnet_teacher": self.bundle.scorer_teacher is not None,
            "has_real_posenet_teacher": self.bundle.pose_scorer_teacher is not None,
            "allow_mock_scorer_teacher": bool(self.bundle.allow_mock_scorer_teacher),
            "allow_segnet_only_research": bool(self.bundle.allow_segnet_only_research),
            "eval_roundtrip_ste": {
                "schema": "mlx_score_aware_eval_roundtrip_ste.v1",
                "enabled": bool(self.bundle.eval_roundtrip_ste_enabled),
                "surface": (
                    "pr95_bicubic_camera_bilinear_scorer_uint8_ste"
                ),
                "camera_hw": [int(v) for v in self.bundle.eval_roundtrip_camera_hw],
                "applied_before": [
                    "reconstruction_loss",
                    "segnet_student_head_loss",
                    "posenet_student_head_loss",
                ],
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "pose_student_input_preprocess": {
                "schema": "mlx_score_aware_pose_student_input_preprocess.v1",
                "mode": str(self.bundle.pose_student_input_preprocess),
                "differentiable": True,
                "source": (
                    "tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx"
                    if self.bundle.pose_student_input_preprocess == "pr95_yuv6"
                    else "decoded_rgb_nhwc01"
                ),
                "authority": "macos_mlx_research_signal_false_authority",
            },
            "scorer_error_pair_sampling": {
                "schema": "mlx_score_aware_scorer_error_pair_sampling_config.v1",
                "enabled": bool(self._pair_sampling_weights),
                "explicit_weight_count": len(self._pair_sampling_weights),
                "default_weight": float(self._pair_sampling_default_weight),
                "pair_index_domain": (
                    "source_pair_indices"
                    if self.bundle.source_pair_indices is not None
                    else "local_pair_indices"
                ),
                "purpose": (
                    "training_time_xray_error_curriculum_not_loss_weight_or_score_claim"
                ),
                "canonical_authority_surface": (
                    "TrainingArtifact top-level false-authority fields"
                ),
            },
            "loss_part_telemetry": {
                "schema": "mlx_score_aware_loss_part_telemetry.v1",
                "emitted_by_train_step": True,
                "required_when_score_terms_enabled": True,
            },
            "scorer_input_distribution_guard": {
                "schema": "mlx_scorer_input_distribution_guard.v1",
                "enabled": self.bundle.scorer_input_distribution_guard_weight > 0.0,
                "weight": float(
                    self.bundle.scorer_input_distribution_guard_weight
                ),
                "saturation_margin": float(
                    self.bundle.scorer_input_distribution_guard_saturation_margin
                ),
                "temperature": float(
                    self.bundle.scorer_input_distribution_guard_temperature
                ),
                "components": [
                    "decoded_rgb_per_channel_mean",
                    "decoded_rgb_per_channel_std",
                    "decoded_rgb_soft_saturation_mass",
                ],
                "authority": "macos_mlx_training_lagrangian_false_authority",
            },
            "pr95_curriculum_controls": {
                "schema": "mlx_pr95_curriculum_controls.v1",
                "pr95_faithful_curriculum_enabled": (
                    self._pr95_faithful_curriculum_enabled
                ),
                "pr95_muon_policy": self._pr95_muon_policy,
                "pr95_stage_source_weight_amplification_enabled": (
                    self._pr95_stage_source_weight_amplification_enabled
                ),
                "source_faithful_default": "faithful_stage8_only",
                "contest_specific_policies": ["every_stage"],
            },
            "active_curriculum_stage_controls": {
                "schema": "mlx_score_aware_active_curriculum_stage_controls.v1",
                "stage_name": self._active_curriculum_stage_name,
                "epoch": self._active_curriculum_stage_epoch,
                "enable_qat": bool(self._active_curriculum_stage_enable_qat),
                "stage_notified_before_train_step": (
                    self._active_curriculum_stage_epoch is not None
                ),
            },
            "train_time_dual_ascent": self._train_time_dual_ascent.as_metadata(),
            "train_time_section_byte_metrics": {
                "schema": "mlx_train_time_section_byte_metrics.v1",
                "enabled": bool(
                    self.bundle.train_time_section_byte_metrics is not None
                    or callable(
                        getattr(self.model, "train_time_section_byte_metrics", None)
                    )
                ),
                "source": self._last_train_time_section_byte_metric_source,
                "metric_count": len(self._last_train_time_section_byte_metrics),
                "contest_rate_score_per_byte": CONTEST_RATE_SCORE_PER_BYTE,
                "last_metrics": dict(self._last_train_time_section_byte_metrics),
                "authority": "macos_mlx_training_lagrangian_false_authority",
            },
        }
        metadata["decoder_weight_gradient_saliency"] = (
            self.decoder_weight_gradient_saliency_summary()
        )
        return metadata

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a deterministic batch of pair indices (Catalog #229 PV)."""
        import numpy as np

        mx = self._mx
        num_pairs = self.bundle.num_pairs
        size = min(max(1, batch_size), num_pairs)
        rng = np.random.RandomState(seed)
        priority_resolution = self._resolve_priority_local_rows()
        weight_resolution = self._resolve_pair_sampling_weights_local()
        priority = tuple(priority_resolution["priority_local_pair_indices"])
        reserve_random_fill = bool(
            priority and len(priority) < num_pairs and size > 1 and num_pairs > 1
        )
        priority_take = min(size - int(reserve_random_fill), len(priority))
        random_fill_count = size - priority_take
        priority_selected: list[int] = []
        if priority_take:
            offset = int(seed) % len(priority)
            rotated_priority = priority[offset:] + priority[:offset]
            priority_selected = [int(value) for value in rotated_priority[:priority_take]]
        if random_fill_count:
            priority_set = set(priority_selected)
            full_priority_set = set(priority)
            remaining_non_priority = [
                pair_index
                for pair_index in range(num_pairs)
                if pair_index not in full_priority_set
            ]
            remaining = remaining_non_priority or [
                pair_index
                for pair_index in range(num_pairs)
                if pair_index not in priority_set
            ]
            random_fill_count = min(random_fill_count, len(remaining))
            random_probabilities = None
            weight_array = weight_resolution["local_pair_sampling_weights"]
            if weight_array is not None:
                remaining_weights = np.asarray(
                    [float(weight_array[int(pair_index)]) for pair_index in remaining],
                    dtype=np.float64,
                )
                mass = float(remaining_weights.sum())
                if mass > 0.0:
                    random_probabilities = remaining_weights / mass
            random_selected = rng.choice(
                remaining,
                size=random_fill_count,
                replace=False,
                p=random_probabilities,
            ).astype("int32")
            sampled = np.asarray(
                [*priority_selected, *[int(value) for value in random_selected.tolist()]],
                dtype=np.int32,
            )
        else:
            sampled = np.asarray(priority_selected, dtype=np.int32)
        sampling_policy = (
            "priority_pairs_then_weighted_random_fill"
            if priority and weight_resolution["enabled"]
            else "priority_pairs_then_random_fill"
            if priority
            else "scorer_error_weighted_random"
            if weight_resolution["enabled"]
            else "deterministic_random"
        )
        sampled_pair_indices = [int(value) for value in sampled.tolist()]
        sampled_weights = []
        weight_array = weight_resolution["local_pair_sampling_weights"]
        if weight_array is not None:
            sampled_weights = [
                float(weight_array[int(pair_index)]) for pair_index in sampled_pair_indices
            ]
        if self.bundle.source_pair_indices is None:
            source_pair_indices = list(sampled_pair_indices)
            priority_source_pair_indices = list(priority_selected)
            pair_index_alignment_mode = "identity_local_rows_are_source_pairs"
        else:
            source_rows = tuple(int(value) for value in self.bundle.source_pair_indices)
            source_pair_indices = [source_rows[int(value)] for value in sampled_pair_indices]
            priority_source_pair_indices = [
                source_rows[int(value)] for value in priority_selected
            ]
            pair_index_alignment_mode = "local_target_rows_to_source_pair_indices"
        self._last_batch_observability = {
            "schema": "mlx_score_aware_pair_batch_observability.v1",
            "num_pairs": int(num_pairs),
            "requested_batch_size": int(batch_size),
            "actual_batch_size": int(size),
            "seed": int(seed),
            "sampling_policy": sampling_policy,
            "requested_priority_pair_indices": list(
                priority_resolution["requested_priority_pair_indices"]
            ),
            "prioritized_pair_count": len(priority),
            "priority_pair_indices_in_batch": priority_selected,
            "priority_local_pair_indices_in_batch": priority_selected,
            "priority_source_pair_indices_in_batch": priority_source_pair_indices,
            "priority_random_fill_reserved": bool(reserve_random_fill),
            "unresolved_priority_pair_indices": list(
                priority_resolution["unresolved_priority_pair_indices"]
            ),
            "priority_pair_alignment_mode": str(
                priority_resolution["priority_pair_alignment_mode"]
            ),
            "scorer_error_pair_sampling": {
                "schema": "mlx_score_aware_scorer_error_pair_sampling.v1",
                "enabled": bool(weight_resolution["enabled"]),
                "explicit_weight_count": int(
                    weight_resolution["explicit_weight_count"]
                ),
                "default_weight": float(self._pair_sampling_default_weight),
                "min_local_weight": weight_resolution["min_local_weight"],
                "max_local_weight": weight_resolution["max_local_weight"],
                "unresolved_weight_pair_indices": list(
                    weight_resolution["unresolved_weight_pair_indices"]
                ),
                "pair_weight_alignment_mode": str(
                    weight_resolution["pair_weight_alignment_mode"]
                ),
                "sampled_pair_weights": sampled_weights,
                "random_fill_weighted": bool(
                    weight_resolution["enabled"] and random_fill_count > 0
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "random_fill_count": int(random_fill_count),
            "pair_indices": sampled_pair_indices,
            "local_pair_indices": sampled_pair_indices,
            "source_pair_indices": source_pair_indices,
            "pair_index_alignment_mode": pair_index_alignment_mode,
            "pair_index_min": int(sampled.min()) if sampled.size else None,
            "pair_index_max": int(sampled.max()) if sampled.size else None,
            "source_pair_index_min": (
                min(source_pair_indices) if source_pair_indices else None
            ),
            "source_pair_index_max": (
                max(source_pair_indices) if source_pair_indices else None
            ),
            "coverage_fraction": float(size) / float(num_pairs) if num_pairs else 0.0,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        return mx.array(sampled.astype("int32"))

    def batch_observability(self, _batch: Any | None = None) -> Mapping[str, Any] | None:
        """Return the last sampled pair-index batch for telemetry only."""

        observed = getattr(self, "_last_batch_observability", None)
        return dict(observed) if isinstance(observed, Mapping) else None

    def _resolve_priority_local_rows(self) -> Mapping[str, Any]:
        requested = tuple(int(value) for value in self._prioritized_pair_indices)
        if not requested:
            return {
                "requested_priority_pair_indices": (),
                "priority_local_pair_indices": (),
                "unresolved_priority_pair_indices": (),
                "priority_pair_alignment_mode": "no_priority_pairs_requested",
            }
        num_pairs = int(self.bundle.num_pairs)
        if self.bundle.source_pair_indices is None:
            local_rows = tuple(value for value in requested if value < num_pairs)
            unresolved = tuple(value for value in requested if value >= num_pairs)
            return {
                "requested_priority_pair_indices": requested,
                "priority_local_pair_indices": local_rows,
                "unresolved_priority_pair_indices": unresolved,
                "priority_pair_alignment_mode": "identity_priority_pairs_are_local_rows",
            }

        source_to_local = {
            int(source_pair_index): int(local_row)
            for local_row, source_pair_index in enumerate(self.bundle.source_pair_indices)
        }
        local_rows = tuple(
            source_to_local[value] for value in requested if value in source_to_local
        )
        unresolved = tuple(value for value in requested if value not in source_to_local)
        return {
            "requested_priority_pair_indices": requested,
            "priority_local_pair_indices": local_rows,
            "unresolved_priority_pair_indices": unresolved,
            "priority_pair_alignment_mode": "source_priority_pairs_to_local_rows",
        }

    def _normalize_prioritized_pair_indices(
        self,
        prioritized_pair_indices: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if prioritized_pair_indices is None:
            return ()
        out: list[int] = []
        seen: set[int] = set()
        for raw in prioritized_pair_indices:
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "prioritized_pair_indices must contain integer pair indices"
                ) from exc
            if value < 0:
                raise ValueError(
                    f"prioritized_pair_indices must be non-negative; got {value}"
                )
            if value not in seen:
                seen.add(value)
                out.append(value)
        return tuple(out)

    def _normalize_pair_sampling_weights(
        self,
        pair_sampling_weights: Mapping[int, float] | None,
    ) -> dict[int, float]:
        if pair_sampling_weights is None:
            return {}
        if not isinstance(pair_sampling_weights, Mapping):
            raise ValueError(
                "pair_sampling_weights must be a mapping of pair_index -> "
                f"non-negative weight; got {type(pair_sampling_weights).__name__}"
            )
        out: dict[int, float] = {}
        for raw_key, raw_value in pair_sampling_weights.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "pair_sampling_weights keys must be integer pair indices"
                ) from exc
            if key < 0:
                raise ValueError(
                    f"pair_sampling_weights keys must be non-negative; got {key}"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"pair_sampling_weights[{key}] must be a finite float"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"pair_sampling_weights[{key}] must be finite and >= 0; got {value!r}"
                )
            out[key] = value
        return out

    def _normalize_pair_sampling_default_weight(self, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pair_sampling_default_weight must be a finite non-negative float"
            ) from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                "pair_sampling_default_weight must be finite and >= 0; got "
                f"{parsed!r}"
            )
        return parsed

    def _normalize_gradient_multipliers(
        self,
        gradient_multiplier_by_name: Mapping[str, float] | None,
        *,
        bias_gradient_multiplier: float | None,
        output_head_bias_gradient_multiplier: float,
    ) -> dict[str, float]:
        self._bias_gradient_multiplier: float | None = None
        if bias_gradient_multiplier is not None:
            try:
                parsed_bias_multiplier = float(bias_gradient_multiplier)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "bias_gradient_multiplier must be None or a finite "
                    "non-negative float"
                ) from exc
            if not math.isfinite(parsed_bias_multiplier) or parsed_bias_multiplier < 0.0:
                raise ValueError(
                    "bias_gradient_multiplier must be finite and >= 0; got "
                    f"{parsed_bias_multiplier!r}"
                )
            self._bias_gradient_multiplier = parsed_bias_multiplier
        try:
            output_bias_multiplier = float(output_head_bias_gradient_multiplier)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "output_head_bias_gradient_multiplier must be a finite "
                "non-negative float"
            ) from exc
        if not math.isfinite(output_bias_multiplier) or output_bias_multiplier < 0.0:
            raise ValueError(
                "output_head_bias_gradient_multiplier must be finite and >= 0; "
                f"got {output_bias_multiplier!r}"
            )
        out: dict[str, float] = {
            "head_rgb_0.bias": output_bias_multiplier,
            "head_rgb_1.bias": output_bias_multiplier,
        }
        if gradient_multiplier_by_name is not None:
            if not isinstance(gradient_multiplier_by_name, Mapping):
                raise ValueError(
                    "gradient_multiplier_by_name must be a mapping of exact "
                    "parameter_name -> finite non-negative multiplier"
                )
            for raw_key, raw_value in gradient_multiplier_by_name.items():
                key = str(raw_key)
                if not key:
                    raise ValueError("gradient_multiplier_by_name keys must be non-empty")
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"gradient_multiplier_by_name[{key!r}] must be a finite float"
                    ) from exc
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"gradient_multiplier_by_name[{key!r}] must be finite "
                        f"and >= 0; got {value!r}"
                    )
                out[key] = value
        return out

    def _gradient_multiplier_for_name(self, name: str) -> float:
        if name in self._gradient_multiplier_by_name:
            return float(self._gradient_multiplier_by_name[name])
        if self._bias_gradient_multiplier is not None and str(name).endswith(".bias"):
            return float(self._bias_gradient_multiplier)
        return 1.0

    def _apply_gradient_multipliers(self, grads: Any) -> tuple[Any, dict[str, float]]:
        """Apply exact-name gradient multipliers before clipping/update."""

        from mlx.utils import tree_flatten, tree_unflatten

        flat: list[tuple[str, Any]] = []
        applied_count = 0
        zeroed_count = 0
        min_multiplier: float | None = None
        max_multiplier: float | None = None
        sum_multiplier = 0.0
        for raw_name, leaf in tree_flatten(grads):
            name = _tree_name_to_saliency_group(raw_name)
            multiplier = self._gradient_multiplier_for_name(name)
            if multiplier != 1.0:
                applied_count += 1
                if multiplier == 0.0:
                    zeroed_count += 1
                leaf = leaf * multiplier
                min_multiplier = (
                    multiplier
                    if min_multiplier is None
                    else min(min_multiplier, multiplier)
                )
                max_multiplier = (
                    multiplier
                    if max_multiplier is None
                    else max(max_multiplier, multiplier)
                )
                sum_multiplier += multiplier
            flat.append((raw_name, leaf))
        metrics = {
            "gradient_multiplier_active": float(applied_count > 0),
            "gradient_multiplier_applied_leaf_count": float(applied_count),
            "gradient_multiplier_zeroed_leaf_count": float(zeroed_count),
            "gradient_multiplier_min": float(
                1.0 if min_multiplier is None else min_multiplier
            ),
            "gradient_multiplier_max": float(
                1.0 if max_multiplier is None else max_multiplier
            ),
            "gradient_multiplier_mean_applied": float(
                1.0 if applied_count == 0 else sum_multiplier / applied_count
            ),
            "gradient_multiplier_output_head_bias": float(
                self._gradient_multiplier_by_name.get("head_rgb_1.bias", 1.0)
            ),
            "gradient_multiplier_bias": (
                -1.0
                if self._bias_gradient_multiplier is None
                else float(self._bias_gradient_multiplier)
            ),
        }
        return tree_unflatten(flat), metrics

    def _resolve_pair_sampling_weights_local(self) -> Mapping[str, Any]:
        if not self._pair_sampling_weights:
            return {
                "enabled": False,
                "explicit_weight_count": 0,
                "local_pair_sampling_weights": None,
                "unresolved_weight_pair_indices": (),
                "pair_weight_alignment_mode": "no_pair_sampling_weights_requested",
                "min_local_weight": None,
                "max_local_weight": None,
            }
        import numpy as np

        num_pairs = int(self.bundle.num_pairs)
        weights = np.full(
            num_pairs,
            float(self._pair_sampling_default_weight),
            dtype=np.float64,
        )
        unresolved: list[int] = []
        if self.bundle.source_pair_indices is None:
            for pair_index, weight in self._pair_sampling_weights.items():
                if pair_index < num_pairs:
                    weights[int(pair_index)] = float(weight)
                else:
                    unresolved.append(int(pair_index))
            alignment = "identity_weight_pairs_are_local_rows"
        else:
            source_to_local = {
                int(source_pair_index): int(local_row)
                for local_row, source_pair_index in enumerate(
                    self.bundle.source_pair_indices
                )
            }
            for pair_index, weight in self._pair_sampling_weights.items():
                local = source_to_local.get(int(pair_index))
                if local is None:
                    unresolved.append(int(pair_index))
                else:
                    weights[int(local)] = float(weight)
            alignment = "source_weight_pairs_to_local_rows"
        if float(weights.sum()) <= 0.0:
            raise ValueError(
                "pair_sampling_weights plus pair_sampling_default_weight leave "
                "zero sampling mass for every local pair; increase at least one "
                "weight or the default."
            )
        return {
            "enabled": True,
            "explicit_weight_count": len(self._pair_sampling_weights),
            "local_pair_sampling_weights": weights,
            "unresolved_weight_pair_indices": tuple(unresolved),
            "pair_weight_alignment_mode": alignment,
            "min_local_weight": float(weights.min()) if weights.size else None,
            "max_local_weight": float(weights.max()) if weights.size else None,
        }

    def _build_wave_n11_optimizer(self, learning_rate: float) -> Any:
        """Build the canonical Wave N+11 stabilizer-aware optimizer.

        This helper builds single-object MLX optimizers only. The default
        ``pact_muon_adamw`` path is intentionally handled in ``train_step`` so
        PR95-derived Muon+AdamW parameter partitioning is preserved; callers
        that want the explicit AdamW control pass ``optimizer_kind="adamw"``.

        Schedule composition:

        - ``warmup_epochs > 0`` AND ``cosine_decay_enabled=True``: composes
          ``linear_schedule(0.0, lr, warmup_steps) >> cosine_decay(lr,
          decay_steps, lr*min_ratio)`` via ``join_schedules``. Matches the
          proven L2 stability hardening pattern.
        - ``warmup_epochs > 0`` AND ``cosine_decay_enabled=False``: linear
          warmup ONLY (constant after warmup completes; canonical for short
          MLX-LOCAL runs where cosine decay overshoots).
        - ``warmup_epochs == 0``: constant lr (legacy behavior).

        Optimizer kind:

        - ``"adamw"``: ``mlx.optimizers.AdamW(learning_rate=sched,
          weight_decay=wd)`` where ``wd`` defaults to AdamW's own 0.01 if
          ``weight_decay`` is None at construction.
        - ``"adam"`` / ``"adamax"`` / ``"adagrad"`` / ``"adadelta"``:
          native MLX optimizer primitives exposed as explicit campaign cells
          so Mac-local sweeps can test optimizer geometry instead of assuming
          AdamW dominance. These MLX classes do not accept weight_decay, so the
          adapter rejects that combination instead of silently dropping decay
          pressure.
        - ``"rmsprop"``: ``mlx.optimizers.RMSprop(learning_rate=sched)`` (MLX
          RMSprop does not accept weight_decay; a future canonical add lands
          weight-decay-via-AdamW-style decoupling per Loshchilov+Hutter 2019
          §4 if Mamba-2 empirically benefits from it).
        - ``"sgd"``: native ``mlx.optimizers.SGD`` with optional explicit
          weight_decay. This is low-priority for score-aware carrier fitting
          but kept available as a curvature/baseline control.
        - ``"lion"``: ``mlx.optimizers.Lion(learning_rate=sched)`` with
          optional explicit weight decay. Native MLX primitive, lower optimizer
          state than Adam-class methods.
        - ``"adafactor"``: ``mlx.optimizers.Adafactor(learning_rate=sched,
          relative_step=False, scale_parameter=False)`` so the caller's
          stage/curriculum LR remains the sole scheduler authority.
        - ``"aurora_like"``: local MLX port of Aurora's leverage-uniform polar
          update for rank >= 2 leaves; rank < 2 leaves use AdamW-style moments.
          This is false-authority timing-smoke support, not PR95 source
          authority.
        - ``"muon"``: native ``mlx.optimizers.Muon`` as an explicit
          all-parameter optimizer-object comparison row. It is intentionally
          separate from Pact's default partitioned Muon+AdamW path.
        - ``"pact_muon_adamw"`` is intentionally NOT built here: it uses the
          PR95-derived partitioned Muon+AdamW helper inside ``train_step`` so
          latents, heads, biases, and scalar-like params never receive a naive
          all-parameter Muon update.
        """
        mlx_optim = self._mlx_optim
        warmup_epochs = self._wave_n11_warmup_epochs
        warmup_steps_per_epoch = self._wave_n11_warmup_steps_per_epoch
        if warmup_epochs > 0:
            warmup_steps = max(1, warmup_epochs * warmup_steps_per_epoch)
            warmup_sched = mlx_optim.linear_schedule(
                0.0, float(learning_rate), warmup_steps
            )
            if self._wave_n11_cosine_decay_enabled:
                total_epochs = int(self._wave_n11_cosine_decay_total_epochs or 0)
                decay_epochs = max(1, total_epochs - warmup_epochs)
                decay_steps = max(1, decay_epochs * warmup_steps_per_epoch)
                end_lr = float(learning_rate) * float(
                    self._wave_n11_cosine_decay_min_lr_ratio
                )
                decay_sched = mlx_optim.cosine_decay(
                    float(learning_rate), decay_steps, end_lr
                )
                lr_sched = mlx_optim.join_schedules(
                    [warmup_sched, decay_sched], [warmup_steps]
                )
            else:
                lr_sched = warmup_sched
        else:
            lr_sched = float(learning_rate)

        if self._wave_n11_optimizer_kind == "adam":
            return mlx_optim.Adam(learning_rate=lr_sched)
        if self._wave_n11_optimizer_kind == "adamax":
            return mlx_optim.Adamax(learning_rate=lr_sched)
        if self._wave_n11_optimizer_kind == "adagrad":
            return mlx_optim.Adagrad(learning_rate=lr_sched)
        if self._wave_n11_optimizer_kind == "adadelta":
            return mlx_optim.AdaDelta(learning_rate=lr_sched)
        if self._wave_n11_optimizer_kind == "rmsprop":
            return mlx_optim.RMSprop(learning_rate=lr_sched)
        if self._wave_n11_optimizer_kind == "sgd":
            if self._wave_n11_weight_decay is None:
                return mlx_optim.SGD(learning_rate=lr_sched)
            return mlx_optim.SGD(
                learning_rate=lr_sched,
                weight_decay=self._wave_n11_weight_decay,
            )
        if self._wave_n11_optimizer_kind == "lion":
            if self._wave_n11_weight_decay is None:
                return mlx_optim.Lion(learning_rate=lr_sched)
            return mlx_optim.Lion(
                learning_rate=lr_sched,
                weight_decay=self._wave_n11_weight_decay,
            )
        if self._wave_n11_optimizer_kind == "adafactor":
            adafactor_kwargs = {
                "learning_rate": lr_sched,
                "relative_step": False,
                "scale_parameter": False,
            }
            if self._wave_n11_weight_decay is not None:
                adafactor_kwargs["weight_decay"] = self._wave_n11_weight_decay
            return mlx_optim.Adafactor(**adafactor_kwargs)
        if self._wave_n11_optimizer_kind == "aurora_like":
            return _build_aurora_like_mlx_optimizer(
                learning_rate=lr_sched,
                weight_decay=self._wave_n11_weight_decay,
            )
        if self._wave_n11_optimizer_kind == "muon":
            if self._wave_n11_weight_decay is None:
                return mlx_optim.Muon(learning_rate=lr_sched)
            return mlx_optim.Muon(
                learning_rate=lr_sched,
                weight_decay=self._wave_n11_weight_decay,
            )
        if self._wave_n11_optimizer_kind == "pact_muon_adamw":
            raise RuntimeError(
                "pact_muon_adamw is a partitioned train_step optimizer, not a "
                "single MLX optimizer object"
            )

        # adamw default + weight_decay override
        if self._wave_n11_weight_decay is None:
            return mlx_optim.AdamW(learning_rate=lr_sched)
        return mlx_optim.AdamW(
            learning_rate=lr_sched,
            weight_decay=self._wave_n11_weight_decay,
        )

    def wave_n11_stabilizer_summary(self) -> Mapping[str, Any]:
        """Return the Wave N+11 stabilizer telemetry summary.

        Consumed by the landing memo's canonical Provenance bind step + the
        empirical anchor's `empirical_output` payload. Per Catalog #305 max
        observability: every Wave N+11 stabilizer run emits this structured
        record so the operator-facing audit surface stays evidence-faithful.
        """
        grad_history = list(self._wave_n11_grad_norm_history)
        return {
            "schema_version": "mlx_score_aware_wave_n11_stabilizer_summary_v1_20260530",
            "grad_clip_max_norm": self._wave_n11_grad_clip_max_norm,
            "warmup_epochs": self._wave_n11_warmup_epochs,
            "warmup_steps_per_epoch": self._wave_n11_warmup_steps_per_epoch,
            "weight_decay": self._wave_n11_weight_decay,
            "weight_decay_supported_by_optimizer": (
                self._wave_n11_weight_decay_supported
            ),
            "weight_decay_policy": (
                "applied_to_native_optimizer"
                if self._wave_n11_weight_decay is not None
                else "not_requested"
            ),
            "optimizer_kind": self._wave_n11_optimizer_kind,
            "optimizer_provenance": dict(
                _MLX_OPTIMIZER_PROVENANCE_BY_KIND.get(
                    self._wave_n11_optimizer_kind,
                    {
                        "borrowed_from": "unknown",
                        "role": "unknown",
                        "contest_adaptation": "unknown",
                    },
                )
            ),
            "pr95_faithful_muon_adamw_partition_enabled": (
                self._pr95_faithful_curriculum_enabled
            ),
            "pr95_muon_policy": self._pr95_muon_policy,
            "pact_native_muon_adamw_partition_enabled": (
                self._wave_n11_optimizer_kind == "pact_muon_adamw"
            ),
            "pact_native_muon_adamw_last_step_summary": (
                dict(self._pact_muon_adamw_last_step_summary)
                if self._pact_muon_adamw_last_step_summary is not None
                else None
            ),
            "cosine_decay_enabled": self._wave_n11_cosine_decay_enabled,
            "cosine_decay_total_epochs": self._wave_n11_cosine_decay_total_epochs,
            "cosine_decay_min_lr_ratio": self._wave_n11_cosine_decay_min_lr_ratio,
            "step_count": self._wave_n11_step_count,
            "grad_norm_clipped_count": self._wave_n11_clipped_count,
            "grad_norm_history_len": len(grad_history),
            "grad_norm_history_max": (
                max(grad_history) if grad_history else None
            ),
            "grad_norm_history_min": (
                min(grad_history) if grad_history else None
            ),
            "grad_norm_history_mean": (
                sum(grad_history) / len(grad_history) if grad_history else None
            ),
        }

    def decoder_weight_gradient_saliency_summary(self) -> Mapping[str, Any]:
        """Return train-time decoder-weight saliency from real MLX gradients.

        The rows are the diagonal-Fisher group proxy used by the NeRV
        waterfill planner: for each decoder tensor, accumulate
        ``sum(grad ** 2)`` across score-aware train steps and expose
        ``mean_grad_sq`` as the scalar group saliency. This is MLX-local
        research evidence only; exact CPU/CUDA replay remains the authority.
        """

        rows: list[dict[str, Any]] = []
        for name in sorted(self._decoder_grad_sq_sum_by_name):
            samples = int(self._decoder_grad_sample_count_by_name.get(name) or 0)
            numel = int(self._decoder_grad_numel_by_name.get(name) or 0)
            grad_sq_sum = float(self._decoder_grad_sq_sum_by_name[name])
            denom = max(1, samples * max(1, numel))
            saliency = grad_sq_sum / float(denom)
            rows.append(
                {
                    "schema": "mlx_decoder_weight_gradient_saliency_row.v1",
                    "group_name": name,
                    "name": name,
                    "saliency": saliency,
                    "decoder_weight_saliency": saliency,
                    "sum_grad_sq": grad_sq_sum,
                    "sample_count": samples,
                    "numel": numel,
                    "max_abs_grad": float(
                        self._decoder_grad_absmax_by_name.get(name, 0.0)
                    ),
                }
            )
        blockers: list[str] = []
        if not rows:
            blockers.append("decoder_weight_gradient_saliency_no_decoder_rows")
        return {
            "schema": DECODER_GRADIENT_SALIENCY_SCHEMA,
            "collector": (
                "MlxScoreAwareAdapter.train_step value_and_grad "
                "decoder tensor grad_sq accumulator"
            ),
            "authority": "macos_mlx_research_signal_false_authority",
            "selection_target": "decoder_weight_waterfill_saliency_json",
            "row_count": len(rows),
            "rows": rows,
            "saliency_by_name": {
                str(row["group_name"]): float(row["saliency"]) for row in rows
            },
            "blockers": blockers,
        }

    def _accumulate_decoder_weight_gradient_saliency(self, grads: Any) -> None:
        """Accumulate squared MLX gradients for decoder-weight waterfilling."""

        from mlx.utils import tree_flatten

        mx = self._mx
        for raw_name, grad in tree_flatten(grads):
            name = _tree_name_to_saliency_group(raw_name)
            if not _is_decoder_weight_saliency_group(name):
                continue
            shape = getattr(grad, "shape", None)
            if shape is None:
                continue
            numel = 1
            for dim in shape:
                numel *= int(dim)
            if numel <= 0:
                continue
            grad_f32 = grad.astype(mx.float32)
            grad_sq_sum = mx.sum(grad_f32 * grad_f32)
            grad_absmax = mx.max(mx.abs(grad_f32))
            mx.eval(grad_sq_sum, grad_absmax)
            self._decoder_grad_sq_sum_by_name[name] = (
                self._decoder_grad_sq_sum_by_name.get(name, 0.0)
                + float(grad_sq_sum.item())
            )
            self._decoder_grad_sample_count_by_name[name] = (
                self._decoder_grad_sample_count_by_name.get(name, 0) + 1
            )
            self._decoder_grad_numel_by_name[name] = int(numel)
            self._decoder_grad_absmax_by_name[name] = max(
                float(self._decoder_grad_absmax_by_name.get(name, 0.0)),
                float(grad_absmax.item()),
            )

    def loss_fn(
        self,
        model: Any,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style A diagnostic loss (no grad/update); Style B train_step is used.

        Provided for Protocol conformance + sister tooling that wants a pure
        loss read. The canonical helper detects ``train_step`` and bypasses
        this.
        """
        mx = self._mx
        if self._pr95_faithful_curriculum_enabled:
            stage_verdict = self._pr95_curriculum_factory.current_stage_verdict(
                self._pr95_global_epoch
            )
            _total, parts = self._pr95_stage_loss_and_parts(
                batch=batch,
                stage_verdict=stage_verdict,
                model=model,
            )
        else:
            _total, parts = score_aware_loss(
                self.bundle, batch, loss_weights=loss_weights
            )
        out: dict[str, float] = {}
        for name, value in parts.items():
            mx.eval(value)
            out[name] = float(value.item())
        return out

    def _with_dual_ascent_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> dict[str, float]:
        """Observe train-step telemetry once and append active dual prices.

        ``effective_loss_weights`` is applied before the gradient step, so each
        step uses the prices learned from prior telemetry. Observation happens
        after loss-part metrics are populated here, updating prices for the next
        step. This keeps PR95, Pact Muon+AdamW, and native MLX optimizer paths
        on the same closed-loop contract without double-updating.
        """

        out = {str(key): float(value) for key, value in dict(metrics).items()}
        self._add_dual_ascent_metric_aliases(out)
        out.update(self._train_time_dual_ascent.observe(out))
        return out

    def _add_dual_ascent_metric_aliases(self, metrics: dict[str, float]) -> None:
        """Expose active PR95-stage scorer terms under canonical dual keys.

        The shared dual-ascent defaults are intentionally family-neutral:
        ``loss_part_distill`` for SegNet last-frame pressure and
        ``loss_part_pose_score_term`` for PoseNet pair/YUV6 score pressure.
        PR95-stage training computes those terms through source-faithful stage
        losses, but names them by stage surface. Alias them here so one
        controller observes scorer distortion across native, PR95, HiNeRV, and
        SNeRV runs while raw ``loss_part_pose_distill`` remains diagnostic MSE.
        """

        if (
            "loss_part_distill" not in metrics
            and "loss_part_pr95_stage_seg_surrogate" in metrics
            and _active_pr95_stage_metric_weight(
                metrics,
                "loss_part_pr95_stage_effective_seg_weight",
                "loss_part_pr95_stage_distill_weight",
            )
        ):
            metrics["loss_part_distill"] = metrics[
                "loss_part_pr95_stage_seg_surrogate"
            ]
        if (
            "loss_part_pose_distill" not in metrics
            and "loss_part_pr95_stage_pose_surrogate" in metrics
            and _active_pr95_stage_metric_weight(
                metrics,
                "loss_part_pr95_stage_effective_pose_weight",
                "loss_part_pr95_stage_pose_distill_weight",
            )
        ):
            metrics["loss_part_pose_distill"] = metrics[
                "loss_part_pr95_stage_pose_surrogate"
            ]
        if (
            "loss_part_pose_score_term" not in metrics
            and "loss_part_pr95_stage_pose_surrogate" in metrics
            and _active_pr95_stage_metric_weight(
                metrics,
                "loss_part_pr95_stage_effective_pose_weight",
                "loss_part_pr95_stage_pose_distill_weight",
            )
        ):
            metrics["loss_part_pose_score_term"] = metrics[
                "loss_part_pr95_stage_pose_surrogate"
            ]
        if (
            "loss_part_segnet_direct_live_distill" not in metrics
            and "loss_part_pr95_stage_segnet_direct_live_distill" in metrics
            and _active_pr95_stage_metric_weight(
                metrics,
                "loss_part_config_weight_pr95_stage_segnet_direct_live_distill",
                "loss_part_stage_weight_pr95_stage_segnet_direct_live_distill",
            )
            and _active_pr95_stage_metric_weight(
                metrics,
                "loss_part_stage_weight_pr95_stage_segnet_direct_live_distill",
                "loss_part_stage_weight_pr95_stage_segnet_direct_live_distill",
            )
        ):
            metrics["loss_part_segnet_direct_live_distill"] = metrics[
                "loss_part_pr95_stage_segnet_direct_live_distill"
            ]
        if (
            "loss_part_segnet_direct_live_argmax_disagreement" not in metrics
            and "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement"
            in metrics
        ):
            metrics["loss_part_segnet_direct_live_argmax_disagreement"] = metrics[
                "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement"
            ]
        seg_argmax = metrics.get("loss_part_segnet_direct_live_argmax_disagreement")
        pose_score = metrics.get("loss_part_pose_score_term")
        if (
            "loss_part_joint_scorer_proxy_nonrate" not in metrics
            and seg_argmax is not None
        ):
            if pose_score is None:
                metrics["loss_part_joint_scorer_proxy_nonrate"] = (
                    1.0e9 + 100.0 * float(seg_argmax)
                )
                return
            metrics["loss_part_joint_scorer_proxy_nonrate"] = (
                100.0 * float(seg_argmax) + float(pose_score)
            )

    def _train_time_section_byte_metrics(
        self,
        *,
        batch: Any,
        loss_weights: Mapping[str, float],
    ) -> dict[str, float]:
        """Return section-byte telemetry usable by train-time dual ascent.

        This is the portable in-training byte-cap bridge. Substrates may expose
        a cheap receiver-packet or predicted section-byte estimate without
        exporting an archive every step. The controller then prices each byte
        with the fixed upstream waterline. The values remain false-authority
        telemetry until a byte-closed archive/runtime replay proves them.
        """

        callback = self.bundle.train_time_section_byte_metrics
        source = "renderer_bundle_callback"
        if callback is None:
            callback = getattr(self.model, "train_time_section_byte_metrics", None)
            source = "model_hook"
        if not callable(callback):
            self._last_train_time_section_byte_metrics = {}
            self._last_train_time_section_byte_metric_source = None
            return {}

        if source == "renderer_bundle_callback":
            raw = callback(self.model, batch, dict(loss_weights))
        else:
            raw = callback(batch=batch, loss_weights=dict(loss_weights))
        if raw is None:
            self._last_train_time_section_byte_metrics = {}
            self._last_train_time_section_byte_metric_source = source
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(
                "train_time_section_byte_metrics must return a Mapping or None; "
                f"got {type(raw).__name__}"
            )

        metrics: dict[str, float] = {
            "train_time_section_byte_metric_active": 1.0,
            "train_time_section_byte_metric_source_bundle_callback": float(
                source == "renderer_bundle_callback"
            ),
            "train_time_section_byte_metric_source_model_hook": float(
                source == "model_hook"
            ),
        }
        sections: dict[str, float] = {}
        archive_bytes = _nonnegative_float_or_none(
            raw.get("archive_bytes", raw.get("total_archive_bytes"))
        )
        if archive_bytes is not None:
            metrics["train_time_archive_bytes"] = archive_bytes
            metrics["train_time_archive_rate_score"] = (
                archive_bytes * CONTEST_RATE_SCORE_PER_BYTE
            )
        raw_sections = raw.get("section_bytes", raw.get("sections"))
        if isinstance(raw_sections, Mapping):
            for name, value in raw_sections.items():
                parsed = _nonnegative_float_or_none(value)
                if parsed is not None:
                    sections[str(name)] = parsed
        for name, value in raw.items():
            key = str(name)
            if key in {
                "archive_bytes",
                "total_archive_bytes",
                "section_bytes",
                "sections",
                "schema",
                "authority",
            }:
                continue
            parsed = _nonnegative_float_or_none(value)
            if parsed is not None:
                sections[key] = parsed
        for name, value in sorted(sections.items()):
            safe = safe_dual_metric_key(name)
            metrics[f"train_time_section_bytes__{safe}"] = value
            metrics[f"train_time_section_rate_score__{safe}"] = (
                value * CONTEST_RATE_SCORE_PER_BYTE
            )
        metrics["train_time_section_byte_metric_count"] = float(len(sections))
        self._last_train_time_section_byte_metrics = dict(metrics)
        self._last_train_time_section_byte_metric_source = source
        return metrics

    def _effective_wave_n11_learning_rate(self, learning_rate: float) -> float:
        """Return the scalar LR for custom step helpers that cannot take schedules."""

        warmup_epochs = self._wave_n11_warmup_epochs
        warmup_steps_per_epoch = self._wave_n11_warmup_steps_per_epoch
        if warmup_epochs <= 0:
            return float(learning_rate)
        mlx_optim = self._mlx_optim
        mx = self._mx
        warmup_steps = max(1, warmup_epochs * warmup_steps_per_epoch)
        warmup_sched = mlx_optim.linear_schedule(
            0.0, float(learning_rate), warmup_steps
        )
        if self._wave_n11_cosine_decay_enabled:
            total_epochs = int(self._wave_n11_cosine_decay_total_epochs or 0)
            decay_epochs = max(1, total_epochs - warmup_epochs)
            decay_steps = max(1, decay_epochs * warmup_steps_per_epoch)
            end_lr = float(learning_rate) * float(
                self._wave_n11_cosine_decay_min_lr_ratio
            )
            decay_sched = mlx_optim.cosine_decay(
                float(learning_rate), decay_steps, end_lr
            )
            schedule = mlx_optim.join_schedules(
                [warmup_sched, decay_sched], [warmup_steps]
            )
        else:
            schedule = warmup_sched
        value = schedule(mx.array(int(self._wave_n11_step_count)))
        mx.eval(value)
        return float(value.item())

    def optimizer_step(
        self, model: Any, loss: Any, learning_rate: float
    ) -> None:
        """Style A stub; this adapter uses Style B ``train_step``.

        Per CLAUDE.md "Comment-only contracts are FORBIDDEN": this raises so a
        caller cannot silently no-op. The canonical helper detects
        ``train_step`` and never calls this.
        """
        raise NotImplementedError(
            "MlxScoreAwareAdapter uses Style B train_step "
            "(combined value+grad+update for MLX value_and_grad). The "
            "canonical helper prefers train_step when present; this "
            "optimizer_step is a Protocol-conformance stub only."
        )

    def train_step(
        self,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Style B combined value+grad+update (canonical MLX training step).

        Trains the renderer via the canonical ``nn.value_and_grad(self.model,
        ...)`` step. When the real-scorer-bound distillation path is active
        (``bundle.scorer_teacher`` + ``bundle.learnable_student_head`` set), the
        student head's ~20 params train JOINTLY on the SAME score-aware loss via
        a sibling ``mx.value_and_grad`` AdamW step — so the renderer is pulled
        toward what the REAL SegNet rewards (Catalog #164 + C6 IBPS lesson),
        not toward a scorer-blind pixel-cosine.
        """
        mx = self._mx
        mlx_nn = self._mlx_nn
        mlx_optim = self._mlx_optim
        effective_loss_weights = self._train_time_dual_ascent.effective_loss_weights(
            loss_weights
        )
        self._active_loss_weights = dict(effective_loss_weights)

        # PR95-faithful 8-stage Muon+AdamW canonical curriculum opt-in path
        # per CLAUDE.md L14 + L15 + the optimizer stack research memo Option A.
        # When enabled, the per-stage optimizer config is loaded from the
        # canonical PR95FaithfulCurriculumFactory and applied via the
        # canonical apply_pr95_mlx_optimizer_step (which routes Muon-eligible
        # vs AdamW-handled params per the canonical partition_pr95_mlx_parameter_names
        # PR95-faithful split). Sister NS kernel (zeropower_via_newtonschulz5_mlx)
        # is the canonical 1:1 PR95 hnerv_muon source implementation.
        if self._pr95_faithful_curriculum_enabled:
            return self._with_dual_ascent_metrics(
                self._train_step_pr95_faithful_curriculum(
                    batch=batch,
                    learning_rate=learning_rate,
                    loss_weights=effective_loss_weights,
                )
            )
        if self._wave_n11_optimizer_kind == "pact_muon_adamw":
            return self._with_dual_ascent_metrics(
                self._train_step_pact_muon_adamw(
                    batch=batch,
                    learning_rate=learning_rate,
                    loss_weights=effective_loss_weights,
                )
            )

        if self._optimizer is None or self._optimizer_lr != learning_rate:
            # Wave N+11 stabilizer: build optimizer with optional canonical
            # warmup + (optional) cosine-decay schedule and configurable kind
            # (adamw|rmsprop) + weight_decay override. When NO stabilizer
            # kwargs are set, defaults are identical to the pre-Wave-N+11
            # adapter (AdamW(learning_rate=lr) with AdamW default
            # weight_decay=0.01) so sister substrates are byte-stable.
            self._optimizer = self._build_wave_n11_optimizer(learning_rate)
            self._optimizer_lr = learning_rate

        def _loss_fn_inner(model: Any) -> Any:
            # NOTE: score_aware_loss reads bundle.model; the value_and_grad
            # closure differentiates ``self.model`` which IS bundle.model.
            total, _parts = score_aware_loss(
                self.bundle, batch, loss_weights=effective_loss_weights
            )
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
        finite_guard_metrics = _assert_mlx_loss_and_gradients_finite(
            mx,
            loss_value,
            grads,
            context=f"{self.substrate_id}_mlx_score_aware_train_step",
        )
        grads, gradient_multiplier_metrics = self._apply_gradient_multipliers(grads)
        self._accumulate_decoder_weight_gradient_saliency(grads)
        # Wave N+11 stabilizer: apply mlx.optimizers.clip_grad_norm BEFORE
        # optimizer.update so the NaN-at-ep-16-18 gradient-explosion signature
        # cannot propagate into the AdamW second-moment buffers (which then
        # poison subsequent steps). Mamba-2 canonical max_norm=1.0.
        if self._wave_n11_grad_clip_max_norm is not None:
            grads, total_norm = mlx_optim.clip_grad_norm(
                grads, self._wave_n11_grad_clip_max_norm
            )
            mx.eval(total_norm)
            grad_norm_pre_clip = float(total_norm.item())
            self._wave_n11_grad_norm_history.append(grad_norm_pre_clip)
            if grad_norm_pre_clip > self._wave_n11_grad_clip_max_norm:
                self._wave_n11_clipped_count += 1
        self._wave_n11_step_count += 1
        self._optimizer.update(self.model, grads)
        post_update_eval_targets, post_update_metrics = self._post_train_step_update(
            batch
        )
        post_update_metrics.update(
            self._train_time_section_byte_metrics(
                batch=batch,
                loss_weights=effective_loss_weights,
            )
        )
        post_update_metrics.update(
            self._score_aware_loss_part_metrics(
                batch=batch,
                loss_weights=effective_loss_weights,
            )
        )

        # Accumulate the MLX arrays the single trailing mx.eval must realize.
        eval_targets: list[Any] = [
            self.model.parameters(),
            self._optimizer.state,
            *post_update_eval_targets,
        ]

        student_head_eval_targets = self._train_student_heads(
            batch=batch,
            learning_rate=learning_rate,
            loss_weights=effective_loss_weights,
        )
        post_update_metrics.update(self._last_student_head_metrics)
        eval_targets.extend(student_head_eval_targets)

        mx.eval(*eval_targets)
        native_optimizer_metrics = {
            "native_mlx_optimizer_active": 1.0,
            "native_mlx_optimizer_kind_adadelta": float(
                self._wave_n11_optimizer_kind == "adadelta"
            ),
            "native_mlx_optimizer_kind_adafactor": float(
                self._wave_n11_optimizer_kind == "adafactor"
            ),
            "native_mlx_optimizer_kind_adagrad": float(
                self._wave_n11_optimizer_kind == "adagrad"
            ),
            "native_mlx_optimizer_kind_aurora_like": float(
                self._wave_n11_optimizer_kind == "aurora_like"
            ),
            "native_mlx_optimizer_kind_adam": float(
                self._wave_n11_optimizer_kind == "adam"
            ),
            "native_mlx_optimizer_kind_adamax": float(
                self._wave_n11_optimizer_kind == "adamax"
            ),
            "native_mlx_optimizer_kind_adamw": float(
                self._wave_n11_optimizer_kind == "adamw"
            ),
            "native_mlx_optimizer_kind_lion": float(
                self._wave_n11_optimizer_kind == "lion"
            ),
            "native_mlx_optimizer_kind_muon": float(
                self._wave_n11_optimizer_kind == "muon"
            ),
            "native_mlx_optimizer_kind_rmsprop": float(
                self._wave_n11_optimizer_kind == "rmsprop"
            ),
            "native_mlx_optimizer_kind_sgd": float(
                self._wave_n11_optimizer_kind == "sgd"
            ),
            "native_mlx_optimizer_weight_decay": (
                0.0
                if self._wave_n11_weight_decay is None
                else float(self._wave_n11_weight_decay)
            ),
            "native_mlx_optimizer_weight_decay_explicit": float(
                self._wave_n11_weight_decay is not None
            ),
        }
        return self._with_dual_ascent_metrics(
            {
                "total": float(loss_value.item()),
                **finite_guard_metrics,
                **gradient_multiplier_metrics,
                **native_optimizer_metrics,
                **post_update_metrics,
            }
        )

    def _train_step_pact_muon_adamw(
        self,
        *,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Pact-native MLX partitioned Muon+AdamW score-aware train step.

        This is original Pact optimizer integration built on two borrowed pieces:
        PR95's hard-won Muon-vs-AdamW parameter partition rule, and the existing
        MLX Newton-Schulz optimizer step helper. It is deliberately separate from
        ``pr95_faithful_curriculum_enabled``: PR95 reproduction owns the staged
        8-part schedule; this path owns fast local optimizer exploration for the
        HiNeRV/SNeRV score loop while keeping the same no-global-Muon safety.
        """

        mx = self._mx
        mlx_nn = self._mlx_nn
        mlx_optim = self._mlx_optim
        self._active_loss_weights = dict(loss_weights)
        from tac.local_acceleration.pr95_hnerv_mlx import (
            Pr95MlxOptimizerConfig,
            Pr95MlxOptimizerState,
            apply_pr95_mlx_optimizer_step,
        )

        if self._pact_muon_adamw_optimizer_state is None:
            self._pact_muon_adamw_optimizer_state = Pr95MlxOptimizerState()

        def _loss_fn_inner(model: Any) -> Any:
            total, _parts = score_aware_loss(
                self.bundle, batch, loss_weights=loss_weights
            )
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
        finite_guard_metrics = _assert_mlx_loss_and_gradients_finite(
            mx,
            loss_value,
            grads,
            context=f"{self.substrate_id}_pact_muon_adamw_train_step",
        )
        grads, gradient_multiplier_metrics = self._apply_gradient_multipliers(grads)
        self._accumulate_decoder_weight_gradient_saliency(grads)
        if self._wave_n11_grad_clip_max_norm is not None:
            grads, total_norm = mlx_optim.clip_grad_norm(
                grads, self._wave_n11_grad_clip_max_norm
            )
            mx.eval(total_norm)
            grad_norm_pre_clip = float(total_norm.item())
            self._wave_n11_grad_norm_history.append(grad_norm_pre_clip)
            if grad_norm_pre_clip > self._wave_n11_grad_clip_max_norm:
                self._wave_n11_clipped_count += 1
        effective_lr = self._effective_wave_n11_learning_rate(learning_rate)
        weight_decay = (
            0.0
            if self._wave_n11_weight_decay is None
            else float(self._wave_n11_weight_decay)
        )
        config = Pr95MlxOptimizerConfig(
            use_muon=True,
            adamw_lr=effective_lr,
            muon_lr=effective_lr * PACT_MUON_ADAMW_MUON_LR_MULTIPLIER,
            latent_lr_mult=PACT_MUON_ADAMW_LATENT_LR_MULTIPLIER,
            muon_weight_decay=weight_decay,
            adamw_weight_decay=weight_decay,
            grad_clip=self._wave_n11_grad_clip_max_norm,
            grad_clip_muon=self._wave_n11_grad_clip_max_norm,
        )
        step_summary = apply_pr95_mlx_optimizer_step(
            self.model,
            grads,
            self._pact_muon_adamw_optimizer_state,
            config,
        )
        self._pact_muon_adamw_last_step_summary = dict(step_summary)
        self._wave_n11_step_count += 1
        post_update_eval_targets, post_update_metrics = self._post_train_step_update(
            batch
        )
        post_update_metrics.update(
            self._train_time_section_byte_metrics(
                batch=batch,
                loss_weights=loss_weights,
            )
        )
        post_update_metrics.update(
            self._score_aware_loss_part_metrics(
                batch,
                loss_weights=loss_weights,
            )
        )
        student_head_eval_targets = self._train_student_heads(
            batch=batch,
            learning_rate=learning_rate,
            loss_weights=loss_weights,
        )
        post_update_metrics.update(self._last_student_head_metrics)
        mx.eval(
            self.model.parameters(),
            loss_value,
            *post_update_eval_targets,
            *student_head_eval_targets,
        )
        return {
            "total": float(loss_value.item()),
            **finite_guard_metrics,
            **gradient_multiplier_metrics,
            "pact_optimizer_uses_muon": 1.0,
            "pact_muon_tensor_count": float(step_summary["muon_tensor_count"]),
            "pact_adamw_tensor_count": float(step_summary["adamw_tensor_count"]),
            "pact_adamw_learning_rate": float(config.adamw_lr),
            "pact_muon_learning_rate": float(config.muon_lr),
            "pact_muon_lr_multiplier": float(PACT_MUON_ADAMW_MUON_LR_MULTIPLIER),
            **post_update_metrics,
        }

    def _train_step_pr95_faithful_curriculum(
        self,
        *,
        batch: Any,
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        """PR95-faithful 8-stage Muon+AdamW canonical training step (Option A).

        Routes the per-stage optimizer state through the canonical
        ``apply_pr95_mlx_optimizer_step`` (which embeds the canonical
        Muon NS kernel + canonical Muon/AdamW partition + canonical
        per-name routing). Each stage actually uses its declared optimizer
        + hyperparameters per CLAUDE.md "NO FAKE IMPLEMENTATIONS"
        non-negotiable.

        The current stage is derived from ``self._pr95_global_epoch`` via
        the canonical ``PR95FaithfulCurriculumFactory.current_stage_verdict``.
        Stage transitions trigger Muon momentum buffer reset (canonical
        Muon-final-stage-only L15 invariant: pre-stage-8 stages use no
        Muon so the buffer set is empty; stage 8 onwards uses fresh buffers).

        Args:
            batch: MLX array of pair indices (from ``sample_batch``).
            learning_rate: ignored at the canonical Pr95MlxOptimizerConfig
                level (stage-specific lr comes from the canonical descriptor);
                preserved in the API for harness backward-compat per Catalog
                #341 canonical-routing-markers + the canonical
                ``SubstrateLongTrainingAdapter`` Protocol contract.
            loss_weights: still passed to sibling student-head optimizers.
                The renderer objective itself is owned by the PR95 stage
                verdict and routes ``stage_verdict.loss_family`` into
                ``pr95_mlx_stage_scorer_surrogate_loss``.
        """
        mx = self._mx
        mlx_nn = self._mlx_nn
        self._active_loss_weights = dict(loss_weights)
        from tac.local_acceleration.pr95_hnerv_mlx import (
            apply_pr95_mlx_optimizer_step,
        )

        # Load the canonical per-stage verdict for the current global_epoch.
        stage_verdict = self._pr95_curriculum_factory.current_stage_verdict(
            self._pr95_global_epoch
        )
        self._notify_renderer_pr95_stage_verdict(stage_verdict)
        # Stage transition reset for Muon buffers per L15 (Muon-final-stage-only).
        if self._pr95_last_stage_verdict is not None:
            prev_stage = self._pr95_last_stage_verdict.stage_index
            curr_stage = stage_verdict.stage_index
            if prev_stage != curr_stage:
                # Reset Muon buffers on stage transition (canonical:
                # pre-stage-8 stages never populate Muon buffers because
                # use_muon=False routes ALL params through the AdamW branch
                # of apply_pr95_mlx_optimizer_step; stage 8 onwards starts
                # with fresh Muon buffers because pre-stage-8 buffers are
                # by construction empty).
                from tac.local_acceleration.pr95_hnerv_mlx import (
                    Pr95MlxOptimizerState,
                )

                self._pr95_optimizer_state = Pr95MlxOptimizerState(
                    step=self._pr95_optimizer_state.step,
                    muon_buffers={},
                    adamw_m=self._pr95_optimizer_state.adamw_m,
                    adamw_v=self._pr95_optimizer_state.adamw_v,
                )
        self._pr95_last_stage_verdict = stage_verdict

        def _loss_fn_inner(model: Any) -> Any:
            # NOTE: the PR95 stage loss reads bundle.model; the value_and_grad
            # closure differentiates ``self.model`` which IS bundle.model.
            total, _parts = self._pr95_stage_loss_and_parts(
                batch=batch,
                stage_verdict=stage_verdict,
                model=model,
                loss_weights=loss_weights,
            )
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
        finite_guard_metrics = _assert_mlx_loss_and_gradients_finite(
            mx,
            loss_value,
            grads,
            context=f"{self.substrate_id}_pr95_curriculum_train_step",
        )
        grads, gradient_multiplier_metrics = self._apply_gradient_multipliers(grads)
        self._accumulate_decoder_weight_gradient_saliency(grads)

        # Apply ONE canonical Muon+AdamW (or AdamW-only) step. The canonical
        # helper handles Muon NS iteration + Muon/AdamW partition + per-name
        # parameter routing internally. config.use_muon controls whether
        # Muon-eligible params get the NS treatment (stage 8 ON) or fall
        # through to AdamW (stages 1-7 OFF).
        config = stage_verdict.optimizer_config
        _summary = apply_pr95_mlx_optimizer_step(
            self.model,
            grads,
            self._pr95_optimizer_state,
            config,
        )
        post_update_eval_targets, post_update_metrics = self._post_train_step_update(
            batch
        )
        post_update_metrics.update(
            self._train_time_section_byte_metrics(
                batch=batch,
                loss_weights=loss_weights,
            )
        )
        post_update_metrics.update(
            self._pr95_stage_loss_part_metrics(
                batch,
                stage_verdict=stage_verdict,
                loss_weights=loss_weights,
            )
        )
        student_head_eval_targets = self._train_student_heads(
            batch=batch,
            learning_rate=learning_rate,
            loss_weights=loss_weights,
        )
        post_update_metrics.update(self._last_student_head_metrics)

        mx.eval(
            self.model.parameters(),
            loss_value,
            *post_update_eval_targets,
            *student_head_eval_targets,
        )
        return {
            "total": float(loss_value.item()),
            **finite_guard_metrics,
            **gradient_multiplier_metrics,
            "pr95_stage_index": float(stage_verdict.stage_index),
            "pr95_stage_qat_active": float(int(bool(stage_verdict.qat_active))),
            "pr95_stage_uses_muon": float(int(stage_verdict.uses_muon)),
            "pr95_stage_optimizer_use_muon": float(
                int(bool(_summary.get("use_muon")))
            ),
            "pr95_stage_muon_tensor_count": float(
                int(_summary.get("muon_tensor_count") or 0)
            ),
            "pr95_stage_adamw_tensor_count": float(
                int(_summary.get("adamw_tensor_count") or 0)
            ),
            "pr95_stage_cat_lambda": float(stage_verdict.cat_lambda),
            "pr95_stage_cat_sigma": float(stage_verdict.cat_sigma),
            **post_update_metrics,
        }

    def _notify_renderer_pr95_stage_verdict(self, stage_verdict: Any) -> None:
        """Forward PR95 QAT activity to renderer-owned train-time quantizers."""

        renderer_hook = getattr(self.model, "notify_pr95_stage_verdict", None)
        if not callable(renderer_hook):
            return
        renderer_hook(int(self._pr95_global_epoch), stage_verdict)

    def notify_global_epoch(self, global_epoch: int) -> None:
        """Notify the adapter of the current global epoch (PR95 curriculum-aware).

        The canonical long-training harness calls this once per epoch so the
        PR95 curriculum factory can advance the stage index correctly. When
        ``pr95_faithful_curriculum_enabled=False`` the PR95-stage update is a
        no-op (preserves backward compat per the legacy adapter API).

        Substrate-AGNOSTIC renderer epoch hook (DreamerV3 τ-anneal sister wave):
        when the renderer module (``self.model``) exposes its OWN
        ``notify_global_epoch`` method, the adapter forwards the epoch to it.
        This is how a substrate with an epoch-dependent forward parameter
        (e.g. the DreamerV3 RSSM Gumbel-Softmax τ-anneal, which closes the
        "Annealed during training (1.0 → 0.1)" comment-only contract) receives
        the per-epoch tick WITHOUT coupling the shared adapter to any specific
        substrate. Renderers that lack the hook (every other MLX substrate) are
        a silent no-op (backward compat per CLAUDE.md "Beauty, simplicity, and
        developer experience"). Forwarding is best-effort: a renderer hook that
        raises must NOT fail the run (the harness already wraps THIS call in a
        try/except, but we keep the renderer hook isolated so a sister-substrate
        renderer bug cannot poison the PR95-stage update above).
        """
        self._pr95_global_epoch = int(global_epoch)
        renderer_hook = getattr(self.model, "notify_global_epoch", None)
        if callable(renderer_hook):
            try:
                renderer_hook(int(global_epoch))
            except Exception as exc:  # pragma: no cover - defensive isolation
                print(
                    "[MlxScoreAwareAdapter] WARN: renderer notify_global_epoch "
                    f"failed at epoch {global_epoch}: {exc!r}"
                )

    def notify_curriculum_stage(self, global_epoch: int, stage: Any) -> None:
        """Notify the adapter/model of full stage controls before train_step.

        ``notify_global_epoch`` is the legacy PR95 stage-index tick. This hook
        carries the full canonical :class:`CurriculumStage`, including
        ``enable_qat``.  That makes per-stage QAT/quant-noise/projection a real
        train-time control for SNeRV/HiNeRV renderers instead of metadata-only
        schema text.
        """

        self._active_curriculum_stage_epoch = int(global_epoch)
        self._active_curriculum_stage_name = str(getattr(stage, "name", ""))
        self._active_curriculum_stage_enable_qat = bool(
            getattr(stage, "enable_qat", False)
        )
        renderer_hook = getattr(self.model, "notify_curriculum_stage", None)
        if callable(renderer_hook):
            renderer_hook(int(global_epoch), stage)

    def post_optimizer_projection(self, *, epoch: int) -> Mapping[str, Any] | None:
        """Forward optional substrate-owned proximal projection hooks."""

        projection_hook = getattr(self.model, "post_optimizer_projection", None)
        if not callable(projection_hook):
            return None
        report = projection_hook(epoch=int(epoch))
        return report if isinstance(report, Mapping) else None

    def export_state_dict(self, model: Any, path: Path) -> None:
        """Export the model state for checkpointing.

        Two paths:

        1. If the substrate wired ``export_state_dict_fn`` (its MLX->PyTorch
           bridge per Catalog #1251), delegate to it (the promotion path).
        2. Otherwise write a numpy-portable MLX-native checkpoint via the
           canonical bridge serializer ``pack_state_dict_numpy`` (commit
           ``980808776``) so the checkpoint round-trips byte-stably with ZERO
           framework import — sister of the substrate's own numpy-portable
           inflate. This keeps checkpointing functional for any MLX substrate
           while the PyTorch promotion bridge is a later deliverable; the
           checkpoint is non-promotable research signal per Catalog #192.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.bundle.export_state_dict_fn is not None:
            self.bundle.export_state_dict_fn(model, path)
            return
        import numpy as np

        from tac.substrates._shared.numpy_portable_inflate import (
            pack_state_dict_numpy,
        )

        flat: dict[str, np.ndarray] = {}

        def _flatten(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _flatten(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _flatten(f"{prefix}.{i}" if prefix else str(i), v)
            elif hasattr(obj, "shape"):
                flat[prefix] = np.asarray(obj)

        _flatten("", model.parameters())
        # Canonical numpy-portable state_dict blob (no PyTorch / pickle); fp32
        # for checkpoint fidelity (the archive grammar owns fp16 storage).
        blob = pack_state_dict_numpy(flat, dtype="fp32")
        blob_path = path.with_suffix(path.suffix + ".npsd")
        blob_path.write_bytes(blob)

    def import_state_dict(self, model: Any, path: Path) -> None:
        """Import a checkpoint emitted by :meth:`export_state_dict`.

        This is the resume-side sister of the NumPy-portable ``.npsd`` export.
        It deliberately refuses partial or shape-mismatched state so the
        canonical long-training resume path cannot pretend that metadata-only
        restoration recovered real MLX weights.
        """

        require_mlx_for_harness()
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten

        from tac.substrates._shared.numpy_portable_inflate import (
            unpack_state_dict_numpy,
        )

        requested = Path(path)
        state_path = requested
        if not state_path.is_file():
            suffixed = requested.with_suffix(requested.suffix + ".npsd")
            if suffixed.is_file():
                state_path = suffixed
        if not state_path.is_file():
            raise FileNotFoundError(f"MLX checkpoint state file not found: {requested}")
        restored = unpack_state_dict_numpy(state_path.read_bytes())
        current_flat = {
            _tree_name_to_saliency_group(name): value
            for name, value in tree_flatten(model.parameters())
        }
        restored_keys = set(restored)
        current_keys = set(current_flat)
        if restored_keys != current_keys:
            missing = sorted(current_keys - restored_keys)[:10]
            extra = sorted(restored_keys - current_keys)[:10]
            raise ValueError(
                "MLX checkpoint key set mismatch; refusing fake/partial resume "
                f"(missing={missing}, extra={extra})"
            )
        flat_updates: list[tuple[str, Any]] = []
        for name, current in current_flat.items():
            arr = np.asarray(restored[name])
            if tuple(arr.shape) != tuple(current.shape):
                raise ValueError(
                    "MLX checkpoint shape mismatch for "
                    f"{name!r}: checkpoint={arr.shape} model={tuple(current.shape)}"
                )
            flat_updates.append((name, mx.array(arr.astype(arr.dtype, copy=False))))
        model.update(tree_unflatten(flat_updates))

    def export_archive(
        self, model: Any, output_dir: Path
    ) -> tuple[Path, str, int] | None:
        """Export the substrate's numpy-portable archive (0.bin)."""
        if self.bundle.export_archive_fn is None:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        return self.bundle.export_archive_fn(model, output_dir)

    def score_aware_components(
        self, model: Any, batch: Any
    ) -> Mapping[str, float] | None:
        """Per-axis decomposition from the Hinton-distilled scorer surrogate.

        PER_AXIS_DECOMPOSITION GAP FIX 2026-05-28 per Z6-v2 + Hinton + 600-pair
        Contrarian VETO `.omx/research/z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_landed_20260528.md`
        op-routable #4 + CLAUDE.md "Subagent coherence-by-default" hook #1
        sensitivity-map ACTIVE + Catalog #356 AxisDecomposition canonical
        contract sub-surface at the per_epoch_metrics emission boundary.

        The pre-fix behavior returned ``None`` because the legacy reasoning
        (mirrored in `tac.substrates.time_traveler_l5_z6.long_training_adapter`
        + sister adapters) was that the MLX L2 trainer is reconstruction-proxy
        only. With the canonical Hinton-distilled scorer-bound surrogate
        landed via the L2 BOTH-TEACHER-WIRED contract (Catalog #164):

        - ``distill`` (KL T=2.0 on REAL SegNet teacher logits) IS the seg
          axis scorer-bound surrogate gradient signal;
        - ``pose_distill`` (MSE on REAL PoseNet teacher pose) IS the pose
          axis scorer-bound surrogate gradient signal.

        Per the Z6-v2 + Hinton apparatus-level finding (5th cross-family
        parity instance confirming the Hinton-distilled scorer-bound
        gradient as dominant in-training convergence driver), the per-axis
        decomposition gap blocked cross-family seg/pose attribution
        analysis — which IS the canonical downstream-of-in-training
        differentiation surface where sub-0.18 lives.

        The decomposition mapping (faithful to the loss math in
        :func:`tac.substrates._shared.mlx_score_aware.loss.score_aware_loss`):

        =========================  ==========================================
        loss component             AxisDecomposition slot
        =========================  ==========================================
        ``parts["distill"]``       ``seg`` (Hinton-KL on real SegNet teacher)
        ``parts["pose_distill"]``  ``pose`` (MSE on real PoseNet teacher)
        ``parts["recon"]``         ``recon_aux`` (per-pixel; not per-axis
                                   attributable but preserved for telemetry
                                   per Catalog #305 observability surface)
        archive_bytes              0.0 (per-step delta undefined; archive
                                   built post-training via export_archive_fn;
                                   the canonical `compose_score_from_axes`
                                   accepts 0.0 as no-signal per
                                   AxisDecomposition NaN-safe rule)
        =========================  ==========================================

        Backward compat (Catalog #341 Tier-A non-promotable preserved): when
        BOTH ``distillation_weight=0.0`` AND ``pose_distillation_weight=0.0``
        (the legacy pure-reconstruction MLX L2 path), returns ``None`` per
        the original observability-only contract — no synthetic per-axis
        signal is emitted from a scorer-unbound loss.

        Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog
        #127/#192/#317/#341: the emitted per-axis values remain
        non-promotable MLX-research-signal; downstream consumers (e.g.
        cross-family attribution analyzers) MUST honor the ``[macOS-MLX
        research-signal]`` axis_tag stamped on the parent TrainingArtifact's
        canonical Provenance and NEVER promote to ``[contest-CPU]`` /
        ``[contest-CUDA]`` without paired Linux x86_64 + NVIDIA evidence.

        Returns:
            ``None`` when neither scorer surrogate is active (pure-recon
            mode; sister-adapter parity).

            Otherwise a ``Mapping[str, float]`` with keys ``seg`` / ``pose``
            / ``recon_aux`` / ``archive_bytes`` (the canonical 4-key shape
            that maps directly into ``AxisDecomposition`` per Catalog #356
            via ``compose_score_from_axes`` at the downstream cathedral
            ranker boundary; the canonical helper accepts the missing
            ``archive_bytes`` channel as 0.0 no-signal).
        """
        # Pure-reconstruction mode: preserve the legacy None contract so
        # sister-adapter parity is unchanged and no synthetic scorer-unbound
        # per-axis row pollutes per_epoch_metrics. Catalog #341 Tier-A
        # observability-only is unaffected.
        scorer_bound = (
            self.bundle.distillation_weight > 0.0
            or self.bundle.pose_distillation_weight > 0.0
        )
        if not scorer_bound:
            return None

        mx = self._mx
        # Reuse the canonical loss decomposition — single source of truth
        # for per-axis attribution per Catalog #290 ADOPT_CANONICAL.
        _total, parts = score_aware_loss(
            self.bundle,
            batch,
            loss_weights=self._active_loss_weights,
        )
        out: dict[str, float] = {}
        # seg axis: only emit when the SegNet teacher is wired (parts may
        # legitimately omit "distill" when distillation_weight=0).
        if "distill" in parts:
            mx.eval(parts["distill"])
            out["seg"] = float(parts["distill"].item())
        else:
            out["seg"] = 0.0
        # pose axis: only emit when the PoseNet teacher is wired.
        if "pose_distill" in parts:
            mx.eval(parts["pose_distill"])
            out["pose"] = float(parts["pose_distill"].item())
        else:
            out["pose"] = 0.0
        # recon_aux: telemetry-only per-pixel reconstruction component
        # (not per-axis attributable; preserved per Catalog #305
        # observability "decomposable per signal" facet).
        if "recon" in parts:
            mx.eval(parts["recon"])
            out["recon_aux"] = float(parts["recon"].item())
        # archive_bytes: per-step delta undefined at MLX L2 (archive built
        # post-training); emit 0.0 per AxisDecomposition NaN-safe rule.
        out["archive_bytes"] = 0.0
        return out


__all__ = [
    "AURORA_LIKE_SOURCE_COMMIT",
    "AURORA_LIKE_SOURCE_REPO",
    "DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND",
    "MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS",
    "SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS",
    "MlxScoreAwareAdapter",
]
