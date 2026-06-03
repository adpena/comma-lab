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

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
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
    "rmsprop",
    "sgd",
    "lion",
    "adafactor",
    "pact_muon_adamw",
)
MLX_SCORE_AWARE_WEIGHT_DECAY_OPTIMIZER_KINDS: tuple[str, ...] = (
    "adamw",
    "sgd",
    "lion",
    "adafactor",
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
        optimizer_kind: str = "adamw",
        cosine_decay_enabled: bool = False,
        cosine_decay_total_epochs: int | None = None,
        cosine_decay_min_lr_ratio: float = 1e-2,
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
                legacy default-on AdamW behavior (backward compat per
                CLAUDE.md "Beauty, simplicity, and developer experience").
                When True, ``train_step`` routes per-stage optimizer state
                through the canonical ``apply_pr95_mlx_optimizer_step`` via
                the canonical ``PR95FaithfulCurriculumFactory``.
            pr95_curriculum_total_epochs: total epoch budget for the PR95
                curriculum; defaults to the canonical 29,650 per L14.
                Required when ``pr95_faithful_curriculum_enabled=True``;
                ignored otherwise.
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
                "adamw" preserves legacy. All supported values route to real
                ``mlx.optimizers`` classes on Apple silicon. "lion" and
                "muon" are native MLX implementations of published algorithms,
                not Apple-specific algorithms; Adafactor is pinned to
                explicit-LR mode so stage curricula remain the authority.
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
                    }
                }
            )
        recon_stage_weight = component_loss_weight(loss_weights, "recon")
        segnet_stage_weight = component_loss_weight(loss_weights, "distill")
        pose_stage_weight = component_loss_weight(loss_weights, "pose_distill")
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
            elif name == "pose_distill":
                out["loss_part_weighted_pose_distill"] = (
                    float(self.bundle.pose_distillation_weight)
                    * pose_stage_weight
                    * scalar
                )
                out["loss_part_stage_weight_pose_distill"] = pose_stage_weight
            elif name == "recon":
                out["loss_part_weighted_recon"] = recon_stage_weight * scalar
                out["loss_part_stage_weight_recon"] = recon_stage_weight
            elif name in weights:
                out[f"loss_part_weighted_{name}"] = float(weights[name]) * scalar
        out["score_aware_loss_parts_active"] = float(
            ("distill" in parts) or ("pose_distill" in parts)
        )
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

            def _head_loss_fn(head_params: Mapping[str, Any]) -> Any:
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
                return (
                    float(self.bundle.distillation_weight)
                    * segnet_stage_weight
                    * distill
                )

            head_params = {"weight": head.weight, "bias": head.bias}
            _hloss, hgrads = mx.value_and_grad(_head_loss_fn)(head_params)
            self._head_optimizer.update(head_params, hgrads)
            head.weight = head_params["weight"]
            head.bias = head_params["bias"]
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
                    decode_frames_nhwc01,
                    pose_student_inputs_nhwc,
                )
                from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                    pose_distillation_mse_loss,
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
                pose_distill = pose_distillation_mse_loss(
                    student_pose=student_pose,
                    teacher_pose=teacher_pose,
                    per_dim_scale=getattr(
                        self.bundle.pose_scorer_teacher,
                        "per_dim_scale",
                        None,
                    ),
                )
                return (
                    float(self.bundle.pose_distillation_weight)
                    * pose_stage_weight
                    * pose_distill
                )

            pose_params = {"weight": pose_head.weight, "bias": pose_head.bias}
            _ploss, pgrads = mx.value_and_grad(_pose_head_loss_fn)(pose_params)
            self._pose_head_optimizer.update(pose_params, pgrads)
            pose_head.weight = pose_params["weight"]
            pose_head.bias = pose_params["bias"]
            eval_targets.extend(
                [pose_head.weight, pose_head.bias, self._pose_head_optimizer.state]
            )

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
            "loss_part_telemetry": {
                "schema": "mlx_score_aware_loss_part_telemetry.v1",
                "emitted_by_train_step": True,
                "required_when_score_terms_enabled": True,
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
        sampled = rng.choice(num_pairs, size=size, replace=False)
        return mx.array(sampled.astype("int32"))

    def _build_wave_n11_optimizer(self, learning_rate: float) -> Any:
        """Build the canonical Wave N+11 stabilizer-aware optimizer.

        When NO stabilizer kwargs are set at construction, this returns
        ``AdamW(learning_rate=lr)`` exactly as the pre-Wave-N+11 code path
        did so sister substrates are byte-stable. When stabilizer kwargs ARE
        set, builds the canonical optimizer with the requested schedule +
        weight_decay + kind.

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
        _total, parts = score_aware_loss(
            self.bundle, batch, loss_weights=loss_weights
        )
        out: dict[str, float] = {}
        for name, value in parts.items():
            mx.eval(value)
            out[name] = float(value.item())
        return out

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
        self._active_loss_weights = dict(loss_weights)

        # PR95-faithful 8-stage Muon+AdamW canonical curriculum opt-in path
        # per CLAUDE.md L14 + L15 + the optimizer stack research memo Option A.
        # When enabled, the per-stage optimizer config is loaded from the
        # canonical PR95FaithfulCurriculumFactory and applied via the
        # canonical apply_pr95_mlx_optimizer_step (which routes Muon-eligible
        # vs AdamW-handled params per the canonical partition_pr95_mlx_parameter_names
        # PR95-faithful split). Sister NS kernel (zeropower_via_newtonschulz5_mlx)
        # is the canonical 1:1 PR95 hnerv_muon source implementation.
        if self._pr95_faithful_curriculum_enabled:
            return self._train_step_pr95_faithful_curriculum(
                batch=batch,
                learning_rate=learning_rate,
                loss_weights=loss_weights,
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
                self.bundle, batch, loss_weights=loss_weights
            )
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
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
            self._score_aware_loss_part_metrics(
                batch,
                loss_weights=loss_weights,
            )
        )

        # Accumulate the MLX arrays the single trailing mx.eval must realize.
        eval_targets: list[Any] = [
            self.model.parameters(),
            self._optimizer.state,
            *post_update_eval_targets,
        ]

        eval_targets.extend(
            self._train_student_heads(
                batch=batch,
                learning_rate=learning_rate,
                loss_weights=loss_weights,
            )
        )

        mx.eval(*eval_targets)
        return {"total": float(loss_value.item()), **post_update_metrics}

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
            loss_weights: passed through to ``score_aware_loss``. For stages
                whose canonical PR95 verdict has ``cat_lambda > 0``, the loss
                also adds the real C1a-style soft categorical entropy term over
                decoder weights using that stage's ``cat_sigma``.
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
            # NOTE: score_aware_loss reads bundle.model; the value_and_grad
            # closure differentiates ``self.model`` which IS bundle.model.
            total, _parts = score_aware_loss(
                self.bundle, batch, loss_weights=loss_weights
            )
            if float(stage_verdict.cat_lambda) > 0.0:
                from tac.substrates._shared.mlx_score_aware.coder_qat import (
                    CoderAwareQATConfig,
                    build_decoder_c1a_entropy_term,
                )

                c1a_entropy = build_decoder_c1a_entropy_term(
                    model,
                    CoderAwareQATConfig(enabled=True, quant_bits=8),
                    sigma=float(stage_verdict.cat_sigma),
                )
                total = total + float(stage_verdict.cat_lambda) * c1a_entropy
            return total

        loss_and_grad_fn = mlx_nn.value_and_grad(self.model, _loss_fn_inner)
        loss_value, grads = loss_and_grad_fn(self.model)
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
        if float(stage_verdict.cat_lambda) > 0.0:
            from tac.substrates._shared.mlx_score_aware.coder_qat import (
                CoderAwareQATConfig,
                build_decoder_c1a_entropy_term,
            )

            c1a_metric = build_decoder_c1a_entropy_term(
                self.model,
                CoderAwareQATConfig(enabled=True, quant_bits=8),
                sigma=float(stage_verdict.cat_sigma),
            )
            mx.eval(c1a_metric)
            c1a_value = float(c1a_metric.item())
            post_update_metrics["loss_part_pr95_c1a_entropy"] = c1a_value
            post_update_metrics["loss_part_weighted_pr95_c1a_entropy"] = (
                float(stage_verdict.cat_lambda) * c1a_value
            )

        mx.eval(
            self.model.parameters(),
            loss_value,
            *post_update_eval_targets,
            *student_head_eval_targets,
        )
        return {
            "total": float(loss_value.item()),
            "pr95_stage_index": float(stage_verdict.stage_index),
            "pr95_stage_uses_muon": float(int(stage_verdict.uses_muon)),
            "pr95_stage_cat_lambda": float(stage_verdict.cat_lambda),
            "pr95_stage_cat_sigma": float(stage_verdict.cat_sigma),
            **post_update_metrics,
        }

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
    "MlxScoreAwareAdapter",
]
