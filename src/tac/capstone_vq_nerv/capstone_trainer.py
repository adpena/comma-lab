# SPDX-License-Identifier: MIT
"""The capstone score-aware trainer (Task #78) — VQ-NeRV + FiLM-pose, #82 loop.

This wraps the verified #82 score-aware mechanism (MLX decoder render -> torch
frozen scorer -> pixel cotangent -> ``mx.vjp`` -> Muon/AdamW) for the ORIGINAL
VQ-NeRV + FiLM-pose bundle. The two additions over the #82 ``MlxScoreAwareTrainer``:

1. **FiLM-pose forward**: the bundle's forward takes ``(indices, pose)``; the
   trainer threads the STORED GT pose (the same scalars the archive stores) into
   both the render and the ``mx.vjp`` traced forward, so the gradient flows
   through the FiLM MLP. The pose half of the score-aware objective re-anchors
   the FiLM to render pose-correct frames.

2. **VQ EMA + commitment**: after each gradient step, the VQ codebook is
   EMA-updated from the most-recent forward (van den Oord §3.2). The commitment
   loss is added to the score-aware Lagrangian (it pulls ``z_e`` toward the
   nearest code so the straight-through gradient is well-conditioned).

The decisive observables are BOTH halves of the joint objective on the EXACT
scorer: ``exact_d_seg`` (the SegNet argmax-disagreement, must DESCEND toward
~5.6e-4) AND ``mean_d_pose`` (the PoseNet MSE, must stay LOW / HOLD because the
FiLM injects the stored pose). A working capstone moves both at the target bytes.

NO MPS; the scorer is frozen torch-CPU (the exact authority); GT only via
``frame_utils.yuv420_to_rgb`` (provided through the bridge's targets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.local_acceleration.pr95_hnerv_mlx import (
    Pr95MlxOptimizerConfig,
    Pr95MlxOptimizerState,
    apply_pr95_mlx_optimizer_step,
    build_parameter_group_lr_policy_fingerprint,
    pr95_mlx_parameter_shape_records,
)
from tac.mlx_pr95_port.curriculum_mechanisms import (
    StageMechanisms,
    add_c1a_entropy_gradient,
    apply_stage_weight_transforms,
    weight_tensor_keys,
)

if TYPE_CHECKING:
    from tac.capstone_vq_nerv.vq_nerv_bundle import CapstoneVqNervBundle
    from tac.mlx_pr95_port.curriculum import StageSpec
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.capstone_vq_nerv.capstone_trainer requires mlx.")


class _CapstoneWeightEMA:
    """Param-wise EMA shadow over the bundle's TRAINABLE parameter tree (audit [A1]).

    Ported from ``tac.mlx_pr95_port.mlx_trainer._MlxEMA``, but the shadow is taken
    over ``bundle.trainable_parameters()`` (decoder + per-frame FiLM weights) — NOT
    ``bundle.parameters()`` — because the VQ codebook has its OWN van-den-Oord EMA
    (``bundle.ema_update_from_last()``) and is a plain array (not a trainable leaf).
    The weight-EMA must not double-average the codebook. This is the EMA
    non-negotiable for the capstone: the inference/archive bytes come from the EMA
    shadow (the averaged, lower-variance, lower-d_seg point), never the live
    final-step weights. The shadow is applied at eval (snapshot+restore) and at
    EXPORT (the trainer's ``export_render_weights`` snapshot-restore contract).
    """

    def __init__(self, bundle: Any, decay: float) -> None:
        _require_mlx()
        self.decay = float(decay)
        # [B4-FIX] number of EMA updates so far — drives the warmup decay below.
        self._num_updates = 0
        self.shadow = {
            k: mx.array(v) for k, v in tree_flatten(bundle.trainable_parameters())
        }

    def effective_decay(self) -> float:
        """[B4-FIX] Warmup decay so the shadow TRACKS the live weights early.

        Delegates to the ONE canonical warmup schedule
        (:func:`tac.ema_warmup.warmup_ema_decay`) so the fix lives in a single
        place across every weight-EMA in the repo (operator 2026-06-11: *"the
        source MLX port is the source of the poison, all must be fixed, reuse as
        much as possible"*). See that module for the full bug writeup; the short
        version: a constant decay (0.997/0.999) lags ~333/1000 STEPS, so on a short
        MLX run the shadow stays ~init and ``exact_d_seg``/export read near-init
        weights even though the live weights solved seg."""
        from tac.ema_warmup import warmup_ema_decay

        return warmup_ema_decay(self._num_updates, self.decay)

    def update(self, bundle: Any) -> None:
        self._num_updates += 1
        d = self.effective_decay()
        for k, v in tree_flatten(bundle.trainable_parameters()):
            if k in self.shadow:
                self.shadow[k] = self.shadow[k] * d + v * (1.0 - d)
            else:
                self.shadow[k] = mx.array(v)
        mx.eval(list(self.shadow.values()))

    @staticmethod
    def _snapshot_live(bundle: Any) -> dict[str, Any]:
        return {k: mx.array(v) for k, v in tree_flatten(bundle.trainable_parameters())}

    def apply_to(self, bundle: Any) -> dict[str, Any]:
        """Install the shadow into the bundle; return the live params to restore."""
        orig = self._snapshot_live(bundle)
        bundle.update(tree_unflatten(list(self.shadow.items())))
        return orig

    @staticmethod
    def restore(bundle: Any, orig: dict[str, Any]) -> None:
        bundle.update(tree_unflatten(list(orig.items())))


@dataclass
class CapstoneTrainConfig:
    """Config for the capstone joint-descent loop (PR95-faithful, C1-C9 fixed)."""

    epochs: int = 60
    batch_size: int = 8
    seg_loss_form: str = "ce_seg_loss"
    seg_weight: float = 100.0
    pose_weight: float = 1.0
    commitment_weight: float = 0.25
    eval_every: int = 10
    seed: int = 0
    # C7: Muon-throughout from epoch 0.
    use_muon: bool = True
    adamw_lr: float = 3.0e-5
    muon_lr: float = 2.0e-4
    latent_lr_mult: float = 10.0
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    grad_clip_muon: float | None = 1.0
    cast_muon_float32_to_bfloat16: bool = True
    # [B1] PR95 per-epoch cosine LR schedule (the dominant d_seg-floor fix). The
    # cosine anneals from 1.0 at epoch 0 to ``eta_min_ratio`` at the final epoch,
    # per stage (a fresh cosine each curriculum stage, like PR95's per-stage
    # optimizer rebuild). ``lr_floor_ratio`` is PR95's ``cfg.lr_floor_ratio``.
    cosine_lr_schedule: bool = True
    lr_floor_ratio: float = 5e-6
    # [A1] Weight-EMA in the capstone (the EMA non-negotiable: eval + EXPORT the
    # shadow, NOT live). [B4]/[B5] default decay 0.997 (the CLAUDE.md-mandated value;
    # 0.999 lags ~1000 steps) and EVAL the shadow by default (the lag is gone at
    # 0.997). The EXPORT path bytes the shadow via ``export_render_weights`` /
    # ``export_codebook`` (the trainer's snapshot-restore-around-export contract).
    ema_decay: float = 0.997
    use_ema_for_eval: bool = True
    # OPT-IN hook to route the pose-FiLM MLP weights to AdamW (not Muon). PR95's
    # Muon class is conv-hidden-weights only; the pose path is a capstone addition.
    # The optimizer-poison audit #3 hypothesized Muon destabilizes the small pose
    # MLP, but the synthetic A/B REFUTED that as a win (Muon-FiLM beat AdamW-FiLM;
    # see .omx/research/quantizr_pose_implementation_audit_*.md §4). DEFAULT FALSE
    # (do not regress validated behavior on an unproven fix); the hook is here for a
    # real-FastViT-PoseNet A/B to decide. The shared partition fn + every other
    # caller are UNTOUCHED; only this substrate can fork the routing of its own
    # FiLM weights via the optimizer-step's additive ``force_adamw_substrings`` hook.
    force_film_to_adamw: bool = False
    # [MLX-GPU] Scorer-loss backend for the per-step gradient. The torch-CPU
    # frozen-scorer bridge (``torch_cpu_bridge``, DEFAULT) is the ~18min/epoch
    # bottleneck on Apple-Silicon (slow_conv2d_forward). ``mlx_gpu`` routes the
    # per-step score-aware FORWARD+BACKWARD through the full MLX SegNet/PoseNet
    # port on the Metal GPU (``MLXGpuScorerBridge``), the throughput unlock for
    # the 600-pair gate. The MLX-GPU loss is a FAST TRAINING SIGNAL ONLY — torch-CPU
    # stays the AUTHORITY: every ``authority_recheck_every`` steps AND every eval,
    # the reported d_seg/d_pose are recomputed on the torch-CPU bridge (the GPU
    # pose drift ~2.76e-4 can swamp a frontier d_pose ~3e-5, so absolute d_pose
    # near the frontier MUST come from torch-CPU). See
    # ``.omx/research/mlx_scorer_port_drift_audit_20260611.md`` + the wire-in memo.
    scorer_backend: str = "torch_cpu_bridge"
    # torch-CPU authority re-score cadence (steps) when ``scorer_backend=mlx_gpu``.
    # 0 disables per-step re-scoring (eval still uses torch-CPU). Used only for
    # telemetry; it does NOT change the gradient (the MLX-GPU loss drives it).
    authority_recheck_every: int = 0
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class CapstoneTrainer:
    """Joint score-aware trainer for the VQ-NeRV + FiLM-pose capstone bundle.

    Args:
        bundle: a :class:`CapstoneVqNervBundle`.
        bridge: a :class:`TorchScorerBridge` (frozen scorer + GT seg/pose targets).
        pose_store: ``(num_pairs, 6)`` float32 stored GT pose (the FiLM carrier —
            this is the SAME tensor the archive stores; the FiLM reads it, the
            export bytes it).
        config: the loop config.
    """

    def __init__(
        self,
        bundle: CapstoneVqNervBundle,
        bridge: TorchScorerBridge,
        pose_store: np.ndarray,
        config: CapstoneTrainConfig,
    ) -> None:
        _require_mlx()
        self.bundle = bundle
        self.bridge = bridge
        self.cfg = config
        self.n_pairs = int(bundle.latents.shape[0])
        self.pose_store = np.asarray(pose_store, dtype=np.float32)
        if self.pose_store.shape[0] != self.n_pairs:
            raise ValueError(
                f"pose_store has {self.pose_store.shape[0]} rows but bundle has "
                f"{self.n_pairs} pairs"
            )
        # Set the FiLM standardization from the stored pose.
        bundle.set_pose_stats(self.pose_store.mean(0), self.pose_store.std(0))
        self.opt_config = Pr95MlxOptimizerConfig(
            use_muon=config.use_muon,
            adamw_lr=config.adamw_lr,
            latent_lr_mult=config.latent_lr_mult,
            muon_lr=config.muon_lr,
            muon_momentum=config.muon_momentum,
            muon_nesterov=config.muon_nesterov,
            muon_ns_steps=config.muon_ns_steps,
            muon_weight_decay=config.muon_weight_decay,
            grad_clip=config.grad_clip,
            grad_clip_muon=config.grad_clip_muon,
            cast_muon_float32_to_bfloat16=config.cast_muon_float32_to_bfloat16,
        )
        self.opt_state = Pr95MlxOptimizerState()
        self._fingerprint = build_parameter_group_lr_policy_fingerprint(
            pr95_mlx_parameter_shape_records(bundle.trainable_parameters())
        )
        # Curriculum weight-domain mechanisms (off by default = single fixed stage).
        self.mechanisms = StageMechanisms()
        self._weight_keys = weight_tensor_keys(bundle.trainable_parameters())
        self._mech_step = 0
        # Pose-FiLM -> AdamW routing (optimizer-poison audit #3): the small zero-init
        # pose MLP belongs in AdamW, not Muon (see CapstoneTrainConfig.force_film_to_adamw).
        self._force_adamw_substrings: tuple[str, ...] | None = (
            ("film",) if config.force_film_to_adamw else None
        )
        # [A1] Weight-EMA over the trainable params (decoder + FiLM); the VQ codebook
        # keeps its own van-den-Oord EMA. Built once; updated after every step.
        self._ema = _CapstoneWeightEMA(self.bundle, config.ema_decay)
        # [B1] cosine-LR schedule state: the base LR for the eta_min floor + the
        # epoch span of the CURRENT stage + the global epoch offset within that
        # stage. ``configure_stage`` resets these per stage (PR95 rebuilds a fresh
        # cosine per stage); ``_current_epoch`` is set by the epoch loop.
        self._cosine_base_lr: float = float(config.adamw_lr)
        self._cosine_total_epochs: int = int(config.epochs)
        self._current_epoch: int = 0
        # [MLX-GPU] Resolve the per-step gradient bridge. torch-CPU bridge is the
        # default + the AUTHORITY for exact_d_seg/mean_d_pose; the MLX-GPU bridge
        # (when selected) drives ONLY the per-step pixel cotangent (fast signal).
        # The MLX-GPU bridge mirrors the torch bridge's exact preprocessing
        # (eval_roundtrip + per-frame resize + rgb_to_yuv6) + the same PR95
        # seg-loss family, so the gradient it produces matches torch-CPU within
        # the measured Metal fp32 reduction-order drift bound.
        self._loss_bridge: Any = self.bridge
        if config.scorer_backend == "mlx_gpu":
            from tac.mlx_pr95_port.mlx_gpu_score_bridge import MLXGpuScorerBridge

            self._loss_bridge = MLXGpuScorerBridge(
                self.bridge.dnet,
                self.bridge.seg_targets_hard,
                self.bridge.pose_targets,
                seg_loss_form=getattr(self.bridge, "seg_loss_form", None)
                or self._resolve_bridge_seg_loss_form(),
                seg_weight=self.bridge.seg_weight,
                pose_weight=self.bridge.pose_weight,
                eval_roundtrip=self.bridge.eval_roundtrip,
                scorer_hw=self.bridge.scorer_hw,
                device_type="gpu",
            )
        elif config.scorer_backend != "torch_cpu_bridge":
            raise ValueError(
                "scorer_backend must be 'torch_cpu_bridge' or 'mlx_gpu', got "
                f"{config.scorer_backend!r}"
            )

    def _resolve_bridge_seg_loss_form(self) -> str:
        """Reverse-map the torch bridge's seg_loss_fn to its canonical form name.

        The torch bridge stores the resolved callable (``seg_loss_fn``), not the
        form name. The MLX-GPU bridge needs the name to pick its sister MLX
        seg-loss. Reverse-lookup the name from the canonical registry; fall back
        to the default ``ce_seg_loss`` if a custom callable was injected.
        """
        from tac.score_aware_loop.live_segnet_loss import STAGE_SEG_LOSS_FNS

        fn = getattr(self.bridge, "seg_loss_fn", None)
        for name, candidate in STAGE_SEG_LOSS_FNS.items():
            if candidate is fn:
                return name
        return "ce_seg_loss"

    def _pose_mx(self, idx_np: np.ndarray) -> Any:
        return mx.array(self.pose_store[idx_np])

    def _render(self, indices: Any, pose: Any) -> Any:
        return self.bundle(indices, pose=pose)

    def _vjp_grads(
        self, indices: Any, pose: Any, pixel_cotangent: Any, commit_cotangent: float
    ) -> dict[str, Any]:
        """Backprop pixel cotangent + commitment gradient through the bundle.

        The traced forward renders the pair (which sets ``last_commitment_loss``)
        and returns ``(render, commit)``; ``mx.vjp`` consumes both with the pixel
        cotangent on the render and ``commit_cotangent`` (= commitment_weight) on
        the scalar commitment loss, so the gradient updates the carrier for BOTH
        the score-aware pixel objective AND the VQ commitment.

        When a curriculum stage has QAT or sigma-noise active, the traced forward
        first applies :func:`apply_stage_weight_transforms` to the (traced) weight
        arrays (STE fake-quant / weight-noise, part of the autodiff graph). The C1a
        entropy gradient (a function of the decoder weights alone) is added AFTER the
        vjp via :func:`add_c1a_entropy_gradient`.
        """
        flat = tree_flatten(self.bundle.trainable_parameters())
        names = [k for k, _ in flat]
        primals = [v for _, v in flat]
        mech = self.mechanisms
        noise_key = None
        if mech.sigma_weight_noise > 0.0:
            noise_key = mx.random.key(self.cfg.seed * 1_000_003 + self._mech_step)

        def forward(*param_arrays: Any) -> list[Any]:
            arrays_by_name = dict(zip(names, param_arrays, strict=True))
            if mech.any_weight_transform:
                arrays_by_name = apply_stage_weight_transforms(
                    arrays_by_name, self._weight_keys, mech, noise_key=noise_key
                )
            self.bundle.update(tree_unflatten(list(arrays_by_name.items())))
            render = self.bundle(indices, pose=pose)
            commit = self.bundle.last_commitment_loss
            return [render, commit]

        cotangents = [pixel_cotangent, mx.array(float(commit_cotangent))]
        _, vjps = mx.vjp(forward, list(primals), cotangents)
        # The traced forward installed the (possibly QAT/noise-transformed) weights
        # into the bundle as a side effect; restore the ORIGINAL primals so the
        # optimizer step updates the un-quantized/un-noised live weights (STE).
        if mech.any_weight_transform:
            self.bundle.update(tree_unflatten(list(zip(names, primals, strict=True))))
        grads = tree_unflatten(list(zip(names, vjps, strict=True)))
        if mech.c1a_active:
            c1a_key = mx.random.key(self.cfg.seed * 7 + self._mech_step)
            grads = add_c1a_entropy_gradient(
                grads,
                self.bundle.trainable_parameters(),
                self._weight_keys,
                mech,
                rng_key=c1a_key,
            )
        self._mech_step += 1
        return grads

    def _lr_scale_for_epoch(self, epoch: int) -> float:
        """[B1] The PR95 cosine multiplier for ``epoch`` within the current stage."""
        if not self.cfg.cosine_lr_schedule:
            return 1.0
        from tac.local_acceleration.pr95_hnerv_mlx import pr95_cosine_lr_scale

        return pr95_cosine_lr_scale(
            epoch,
            self._cosine_total_epochs,
            base_lr=self._cosine_base_lr,
            lr_floor_ratio=self.cfg.lr_floor_ratio,
        )

    def step(self, idx_np: np.ndarray, *, lr_scale: float = 1.0) -> dict[str, Any]:
        """One joint training step. Returns telemetry.

        ``lr_scale`` is the PR95 per-epoch cosine multiplier (audit [B1]); the epoch
        loop computes it once per epoch and threads it to every batch in that epoch.
        """
        _require_mlx()
        indices = mx.array(idx_np.astype(np.int32))
        pose = self._pose_mx(idx_np)
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        render = self._render(indices, pose)
        mx.eval(render, self.bundle.last_commitment_loss)
        # [MLX-GPU] The per-step gradient comes from the loss bridge (torch-CPU by
        # default; MLX-GPU when scorer_backend=mlx_gpu). torch-CPU stays the
        # AUTHORITY for the reported d_seg/d_pose (recomputed at eval + on the
        # authority_recheck_every cadence below).
        result = self._loss_bridge.loss_and_pixel_grad(render, idx_t)
        authority_d_seg = result.d_seg
        if (
            self._loss_bridge is not self.bridge
            and self.cfg.authority_recheck_every > 0
            and (self._mech_step % self.cfg.authority_recheck_every == 0)
        ):
            # torch-CPU authority re-score of THIS batch's d_seg (telemetry only;
            # does NOT change the gradient — the MLX-GPU loss already drove it).
            authority_d_seg = self.bridge.exact_d_seg(render, idx_t)
        grads = self._vjp_grads(
            indices, pose, result.pixel_cotangent, self.cfg.commitment_weight
        )
        summary = apply_pr95_mlx_optimizer_step(
            self.bundle,
            grads,
            self.opt_state,
            self.opt_config,
            parameter_group_fingerprint=self._fingerprint,
            force_adamw_substrings=self._force_adamw_substrings,
            lr_scale=lr_scale,
        )
        # VQ EMA codebook update from the most-recent forward (van den Oord §3.2).
        self.bundle.ema_update_from_last()
        # [A1] Weight-EMA update over the trainable params AFTER the optimizer step.
        self._ema.update(self.bundle)
        mx.eval(self.bundle.parameters())
        return {
            "loss": result.loss_value,
            "seg": result.seg_loss_value,
            "pose": result.pose_loss_value,
            "commit": float(self.bundle.last_commitment_loss),
            "d_seg_batch": result.d_seg,
            "d_seg_batch_authority": authority_d_seg,
            "lr_scale": float(summary.get("lr_scale", lr_scale)),
            "effective_muon_lr": float(summary.get("effective_muon_lr", 0.0)),
            "grad_clip_would_clip": summary.get(
                "gradient_clip_would_clip_count", 0
            ),
        }

    def _resolve_use_ema(self, use_ema: bool | None) -> bool:
        return self.cfg.use_ema_for_eval if use_ema is None else bool(use_ema)

    def exact_d_seg(self, *, use_ema: bool | None = None) -> float:
        """Mean EXACT live-render d_seg over all pairs (pose-FiLM render).

        [A1] When ``use_ema`` resolves True (the default = ``cfg.use_ema_for_eval``),
        the EMA shadow is installed (snapshot+restore) so the reported d_seg is the
        averaged, lower-variance shadow's d_seg — the SAME point the export bytes.
        """
        _require_mlx()
        do_ema = self._resolve_use_ema(use_ema)
        orig = self._ema.apply_to(self.bundle) if do_ema else None
        try:
            total = 0.0
            n = 0
            for start in range(0, self.n_pairs, self.cfg.batch_size):
                idx_np = np.arange(
                    start, min(start + self.cfg.batch_size, self.n_pairs)
                )
                indices = mx.array(idx_np.astype(np.int32))
                render = self._render(indices, self._pose_mx(idx_np))
                mx.eval(render)
                d = self.bridge.exact_d_seg(
                    render, torch.from_numpy(idx_np.astype(np.int64))
                )
                total += d * len(idx_np)
                n += len(idx_np)
            return total / max(n, 1)
        finally:
            if orig is not None:
                _CapstoneWeightEMA.restore(self.bundle, orig)

    def mean_d_pose(self, *, use_ema: bool | None = None) -> float:
        """Mean EXACT PoseNet MSE over all pairs (the pose half of the score).

        Re-measures the PoseNet output on the LIVE pose-FiLM render vs the GT
        pose targets the bridge holds — the same per-pair pose MSE the evaluator
        charges (``d_pose``). This is the decisive POSE observable.

        MEASUREMENT-CONSISTENCY FIX (optimizer-poison audit #4): this routes through
        ``bridge.exact_d_pose``, which applies the SAME eval_roundtrip (bicubic-up
        874x1164 -> bilinear-down 384x512 -> STE/uint8 round) that the LOSS path and
        ``bridge.exact_d_seg`` apply. The prior bespoke ``_exact_d_pose`` did a
        clamp-only measurement (NO uint8 roundtrip), so the reported d_pose
        UNDERSTATED the contest value — uint8 luma quant is exactly where pose
        drifts, and a "pose held" verdict on a clamp-only number could send a
        candidate to a paid eval that misses the tube. The honest measurement uses
        the bridge's roundtrip-consistent path (CLAUDE.md "eval_roundtrip" +
        "Apples-to-apples evidence discipline").
        """
        _require_mlx()
        bridge = self.bridge
        if bridge.pose_targets is None:
            return 0.0
        do_ema = self._resolve_use_ema(use_ema)
        orig = self._ema.apply_to(self.bundle) if do_ema else None
        try:
            total = 0.0
            n = 0
            for start in range(0, self.n_pairs, self.cfg.batch_size):
                idx_np = np.arange(
                    start, min(start + self.cfg.batch_size, self.n_pairs)
                )
                indices = mx.array(idx_np.astype(np.int32))
                render = self._render(indices, self._pose_mx(idx_np))
                mx.eval(render)
                idx_t = torch.from_numpy(idx_np.astype(np.int64))
                d = bridge.exact_d_pose(render, idx_t)
                total += d * len(idx_np)
                n += len(idx_np)
            return total / max(n, 1)
        finally:
            if orig is not None:
                _CapstoneWeightEMA.restore(self.bundle, orig)

    def configure_stage(self, spec: StageSpec, *, optimizer_schedule: str) -> None:
        """Switch to a PR95 curriculum stage (CurriculumTrainerProtocol).

        Sets the bridge seg-loss family, rebuilds the optimizer config (LR +
        Muon-vs-AdamW per the resolved schedule + per-stage grad-clip/wd), and arms
        the per-stage QAT / C1a / sigma-noise mechanisms.

        [B2] PR95 builds a FRESH optimizer + a FRESH cosine scheduler per stage (each
        ``stage{n}.main()`` constructs its own ``AdamW``/``Muon`` + ``LambdaLR``). So
        this RESETS the optimizer step counter (the AdamW bias-correction warmup
        restarts) and the cosine schedule base/span per stage. The WEIGHTS + the
        weight-EMA shadow + the VQ codebook carry across stages (PR95 resumes the
        decoder/latents between stages); only the optimizer MOMENTUM/bias-correction
        and the cosine restart per stage, which is the PR95 inter-stage contract.
        """
        from tac.mlx_pr95_port.curriculum import resolve_use_muon

        self.bridge.set_seg_loss_form(spec.seg_loss_form)
        # [MLX-GPU] keep the loss bridge's stage seg-loss + weights in sync.
        if self._loss_bridge is not self.bridge:
            self._loss_bridge.set_seg_loss_form(spec.seg_loss_form)
            self._loss_bridge.seg_weight = float(spec.seg_weight)
            self._loss_bridge.pose_weight = float(spec.pose_weight)
        self.cfg.seg_weight = spec.seg_weight
        self.cfg.pose_weight = spec.pose_weight
        use_muon = resolve_use_muon(spec, optimizer_schedule)
        self.opt_config = Pr95MlxOptimizerConfig(
            use_muon=use_muon,
            adamw_lr=spec.adamw_lr,
            latent_lr_mult=spec.latent_lr_mult,
            muon_lr=spec.muon_lr,
            muon_momentum=self.cfg.muon_momentum,
            muon_nesterov=self.cfg.muon_nesterov,
            muon_ns_steps=self.cfg.muon_ns_steps,
            muon_weight_decay=spec.muon_weight_decay,
            grad_clip=spec.grad_clip,
            grad_clip_muon=spec.grad_clip_muon,
            cast_muon_float32_to_bfloat16=self.cfg.cast_muon_float32_to_bfloat16,
        )
        # [B2] fresh optimizer state per stage (bias-correction warmup + momentum
        # restart), matching PR95's per-stage optimizer rebuild.
        self.opt_state = Pr95MlxOptimizerState()
        # [B1] fresh cosine schedule per stage: base LR for the eta_min floor +
        # the stage's epoch span. The epoch loop steps the cosine over [0, epochs).
        self._cosine_base_lr = float(spec.adamw_lr)
        self._cosine_total_epochs = int(spec.epochs)
        self._current_epoch = 0
        self.mechanisms = StageMechanisms(
            use_qat=spec.use_qat,
            sigma_weight_noise=spec.sigma_weight_noise,
            cat_lambda=spec.cat_lambda,
            cat_sigma=spec.cat_sigma,
        )
        # [A1] per-stage EMA decay can change with the stage spec (PR95 carries one
        # decay; the StageSpec exposes ema_decay so the curriculum can sweep it).
        self._ema.decay = float(getattr(spec, "ema_decay", self.cfg.ema_decay))

    def run_stage_epochs(self, spec: StageSpec) -> dict[str, Any]:
        """Run ``spec.epochs`` epochs of the configured stage (CurriculumTrainerProtocol)."""
        return self._run_epochs(spec.epochs, stage_name=spec.name)

    def run_curriculum(
        self, stages: Any, *, optimizer_schedule: str, on_stage_done: Any = None
    ) -> Any:
        """Run a full PR95 curriculum on this capstone trainer (convenience wrapper)."""
        from tac.mlx_pr95_port.curriculum import run_curriculum as _run

        np.random.seed(self.cfg.seed)
        mx.random.seed(self.cfg.seed)
        return _run(
            self, stages, optimizer_schedule=optimizer_schedule,
            on_stage_done=on_stage_done,
        )

    def _run_epochs(self, epochs: int, *, stage_name: str | None = None) -> dict[str, Any]:
        """Run ``epochs`` epochs of the joint loop with the CURRENT stage config."""
        _require_mlx()
        cfg = self.cfg
        # [B1] the cosine spans THIS call's epoch count (configure_stage set the
        # stage span; the un-curriculum'd train() sets it to cfg.epochs).
        self._cosine_total_epochs = int(epochs)

        d_seg_initial = self.exact_d_seg(use_ema=False)
        d_pose_initial = self.mean_d_pose(use_ema=False)
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0
        last_lr_scale = 1.0

        for epoch in range(epochs):
            self._current_epoch = epoch
            lr_scale = self._lr_scale_for_epoch(epoch)  # [B1] per-epoch cosine
            last_lr_scale = lr_scale
            perm = np.random.permutation(self.n_pairs)
            ep_seg = ep_loss = ep_commit = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np, lr_scale=lr_scale)
                ep_seg += row["seg"]
                ep_loss += row["loss"]
                ep_commit += row["commit"]
                if row["grad_clip_would_clip"] > 0:
                    clip_would_steps += 1
                total_steps += 1
                nb += 1

            if (epoch + 1) % cfg.eval_every == 0 or epoch == epochs - 1:
                d_seg = self.exact_d_seg()
                d_pose = self.mean_d_pose()
                best_d_seg = min(best_d_seg, d_seg)
                trow = {
                    "epoch": epoch + 1,
                    "stage": stage_name,
                    "exact_d_seg": d_seg,
                    "mean_d_pose": d_pose,
                    "loss_mean": ep_loss / max(nb, 1),
                    "seg_loss_mean": ep_seg / max(nb, 1),
                    "commit_mean": ep_commit / max(nb, 1),
                    "lr_scale": lr_scale,
                    "clip_would_fraction": clip_would_steps / max(total_steps, 1),
                }
                trajectory.append(trow)
                cfg.telemetry.append(trow)

        return {
            "stage": stage_name,
            "epochs": epochs,
            "d_seg_initial": d_seg_initial,
            "d_pose_initial": d_pose_initial,
            "d_seg_final": trajectory[-1]["exact_d_seg"] if trajectory else d_seg_initial,
            "d_pose_final": trajectory[-1]["mean_d_pose"] if trajectory else d_pose_initial,
            "d_seg_best": best_d_seg if best_d_seg != float("inf") else d_seg_initial,
            "lr_scale_final": last_lr_scale,
            "seg_descended": (
                trajectory[-1]["exact_d_seg"] < d_seg_initial - 1e-4
                if trajectory
                else False
            ),
            "trajectory": trajectory,
        }

    def train(self) -> dict[str, Any]:
        """Run the joint loop. Returns the joint-descent summary + trajectory."""
        _require_mlx()
        cfg = self.cfg
        np.random.seed(cfg.seed)
        mx.random.seed(cfg.seed)
        # [B1] the un-curriculum'd run is a single cosine over cfg.epochs; the base
        # LR for the eta_min floor is cfg.adamw_lr (PR95's eta_min denominator).
        self._cosine_base_lr = float(cfg.adamw_lr)
        self._cosine_total_epochs = int(cfg.epochs)

        d_seg_initial = self.exact_d_seg(use_ema=False)
        d_pose_initial = self.mean_d_pose(use_ema=False)
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0
        last_lr_scale = 1.0

        for epoch in range(cfg.epochs):
            self._current_epoch = epoch
            lr_scale = self._lr_scale_for_epoch(epoch)  # [B1] per-epoch cosine
            last_lr_scale = lr_scale
            perm = np.random.permutation(self.n_pairs)
            ep_seg = ep_loss = ep_commit = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np, lr_scale=lr_scale)
                ep_seg += row["seg"]
                ep_loss += row["loss"]
                ep_commit += row["commit"]
                if row["grad_clip_would_clip"] > 0:
                    clip_would_steps += 1
                total_steps += 1
                nb += 1

            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                d_seg = self.exact_d_seg()
                d_pose = self.mean_d_pose()
                best_d_seg = min(best_d_seg, d_seg)
                trow = {
                    "epoch": epoch + 1,
                    "exact_d_seg": d_seg,
                    "mean_d_pose": d_pose,
                    "loss_mean": ep_loss / max(nb, 1),
                    "seg_loss_mean": ep_seg / max(nb, 1),
                    "commit_mean": ep_commit / max(nb, 1),
                    "lr_scale": lr_scale,
                    "clip_would_fraction": clip_would_steps / max(total_steps, 1),
                }
                trajectory.append(trow)
                cfg.telemetry.append(trow)

        return {
            "d_seg_initial": d_seg_initial,
            "d_pose_initial": d_pose_initial,
            "d_seg_final": trajectory[-1]["exact_d_seg"] if trajectory else None,
            "d_pose_final": trajectory[-1]["mean_d_pose"] if trajectory else None,
            "d_seg_best": best_d_seg,
            "lr_scale_final": last_lr_scale,
            "seg_descended": (
                trajectory[-1]["exact_d_seg"] < d_seg_initial - 1e-4
                if trajectory
                else False
            ),
            "pose_held": (
                trajectory[-1]["mean_d_pose"] <= d_pose_initial + 1e-3
                if trajectory
                else False
            ),
            "trajectory": trajectory,
        }

    def export_render_weights(self) -> dict[str, np.ndarray]:
        """[A1] The render-basis weights to EXPORT = the EMA shadow (NOT live).

        The EMA non-negotiable: the archive bytes come from ``ema.state_dict()``.
        This installs the EMA shadow over the bundle's trainable params (decoder +
        FiLM), extracts the full contest-keyed render-weight dict via
        ``full_render_weights_from_bundle`` (the SAME path the live export used), then
        restores the live params. If ``use_ema_for_eval`` is False the caller has
        opted out of the shadow entirely, so the live weights are returned (the
        explicit research-only escape hatch, consistent with the eval path).
        """
        _require_mlx()
        from tac.capstone_vq_nerv.numpy_reference import (
            full_render_weights_from_bundle,
        )

        if not self.cfg.use_ema_for_eval:
            return full_render_weights_from_bundle(self.bundle)
        orig = self._ema.apply_to(self.bundle)
        try:
            return full_render_weights_from_bundle(self.bundle)
        finally:
            _CapstoneWeightEMA.restore(self.bundle, orig)

    def export_stored_latents(self) -> np.ndarray:
        """[A1] The per-pair latents to EXPORT for the ``stored_latent`` carrier.

        The per-pair latents ARE trainable params (in the weight-EMA shadow), so —
        exactly like ``export_render_weights`` — the EXPORTED latents are the EMA
        SHADOW (the averaged, lower-variance point the advisory d_seg/d_pose are
        measured on), NOT the live final-step latents. This keeps the archive bytes
        consistent with the eval/advisory shadow (the EMA non-negotiable applied to
        the carrier). When ``use_ema_for_eval`` is False the caller has opted out of
        the shadow, so the live latents are returned (the research-only escape hatch,
        consistent with the eval + render-weight paths).
        """
        _require_mlx()
        if self.bundle.carrier != "stored_latent":
            raise RuntimeError(
                "export_stored_latents() is only valid for the 'stored_latent' "
                f"carrier; this bundle is '{self.bundle.carrier}'."
            )
        if not self.cfg.use_ema_for_eval:
            return self.bundle.all_latents()
        orig = self._ema.apply_to(self.bundle)
        try:
            return self.bundle.all_latents()
        finally:
            _CapstoneWeightEMA.restore(self.bundle, orig)


__all__ = [
    "CapstoneTrainConfig",
    "CapstoneTrainer",
]
