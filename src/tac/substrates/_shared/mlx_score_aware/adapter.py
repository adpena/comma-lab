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
from tac.substrates._shared.mlx_score_aware.loss import score_aware_loss

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle


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
    ) -> None:
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

    def sample_batch(self, batch_size: int, seed: int) -> Any:
        """Sample a deterministic batch of pair indices (Catalog #229 PV)."""
        import numpy as np

        mx = self._mx
        num_pairs = self.bundle.num_pairs
        size = min(max(1, batch_size), num_pairs)
        rng = np.random.RandomState(seed)
        sampled = rng.choice(num_pairs, size=size, replace=False)
        return mx.array(sampled.astype("int32"))

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
        if self._optimizer is None or self._optimizer_lr != learning_rate:
            self._optimizer = mlx_optim.AdamW(learning_rate=learning_rate)
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
        self._optimizer.update(self.model, grads)

        # Accumulate the MLX arrays the single trailing mx.eval must realize.
        eval_targets: list[Any] = [self.model.parameters(), self._optimizer.state]

        # Sibling SegNet student-head step (real-scorer-bound distillation only).
        head = self.bundle.learnable_student_head
        if (
            self.bundle.distillation_weight > 0.0
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
                # Re-derive the distill term as a pure function of the head
                # params so MLX differentiates the head ONLY (renderer is held
                # via stop_gradient on the decoded frames + the teacher is
                # already gradient-blocked).
                from tac.substrates._shared.mlx_score_aware.loss import (
                    decode_frames_nhwc01,
                )
                from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                    hinton_distilled_kl_t2_loss,
                )

                rgb_0, rgb_1 = decode_frames_nhwc01(self.bundle, batch)
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
                return self.bundle.distillation_weight * hinton_distilled_kl_t2_loss(
                    student_logits=student,
                    teacher_logits=teacher,
                    temperature=self.bundle.distillation_temperature,
                )

            head_params = {"weight": head.weight, "bias": head.bias}
            _hloss, hgrads = mx.value_and_grad(_head_loss_fn)(head_params)
            self._head_optimizer.update(head_params, hgrads)
            # AdamW.update mutates head_params in place via tree semantics; the
            # updated arrays are the optimizer's view, so write them back.
            head.weight = head_params["weight"]
            head.bias = head_params["bias"]
            eval_targets.extend(
                [head.weight, head.bias, self._head_optimizer.state]
            )

        # Sibling POSE student-head step (real-PoseNet-bound distillation only).
        pose_head = self.bundle.learnable_pose_student_head
        if (
            self.bundle.pose_distillation_weight > 0.0
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
                # Re-derive the pose-MSE distill term as a pure function of the
                # pose head params so MLX differentiates the pose head ONLY
                # (renderer held via stop_gradient on the decoded pair; teacher
                # already gradient-blocked).
                from tac.substrates._shared.mlx_score_aware.loss import (
                    decode_frames_nhwc01,
                )
                from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
                    pose_distillation_mse_loss,
                )

                rgb_0, rgb_1 = decode_frames_nhwc01(self.bundle, batch)
                rgb_0 = mx.stop_gradient(rgb_0)
                rgb_1 = mx.stop_gradient(rgb_1)
                student_pose = pose_head.forward_with_params(
                    rgb_0,
                    rgb_1,
                    {
                        "weight": pose_params["weight"],
                        "bias": pose_params["bias"],
                    },
                )
                teacher_pose = mx.stop_gradient(
                    self.bundle.pose_scorer_teacher.teacher_pose_for_indices(batch)
                )
                return self.bundle.pose_distillation_weight * pose_distillation_mse_loss(
                    student_pose=student_pose,
                    teacher_pose=teacher_pose,
                    per_dim_scale=getattr(
                        self.bundle.pose_scorer_teacher,
                        "per_dim_scale",
                        None,
                    ),
                )

            pose_params = {"weight": pose_head.weight, "bias": pose_head.bias}
            _ploss, pgrads = mx.value_and_grad(_pose_head_loss_fn)(pose_params)
            self._pose_head_optimizer.update(pose_params, pgrads)
            pose_head.weight = pose_params["weight"]
            pose_head.bias = pose_params["bias"]
            eval_targets.extend(
                [pose_head.weight, pose_head.bias, self._pose_head_optimizer.state]
            )

        mx.eval(*eval_targets)
        return {"total": float(loss_value.item())}

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
        _total, parts = score_aware_loss(self.bundle, batch)
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
