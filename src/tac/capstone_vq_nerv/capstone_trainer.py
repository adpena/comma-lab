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
    ema_decay: float = 0.999
    use_ema_for_eval: bool = False  # eval LIVE weights (the 0.999-lag landmine).
    # Route the pose-FiLM MLP weights to AdamW (not Muon). Muon's Newton-Schulz
    # orthogonalization gives grad-norm-INDEPENDENT O(1) step magnitudes; for a
    # small zero-init pose MLP this can over/under-shoot the pose basin. PR95's
    # Muon class is conv-hidden-weights only — the pose path is a capstone addition
    # that belongs in AdamW (faithful-core + adapted-synergy; CLAUDE.md
    # "UNIQUE-AND-COMPLETE-PER-METHOD"). The shared partition fn + every other
    # caller are UNTOUCHED; only this substrate forks the routing of its own
    # FiLM weights via the optimizer-step's additive ``force_adamw_substrings`` hook.
    force_film_to_adamw: bool = True
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

    def step(self, idx_np: np.ndarray) -> dict[str, Any]:
        """One joint training step. Returns telemetry."""
        _require_mlx()
        indices = mx.array(idx_np.astype(np.int32))
        pose = self._pose_mx(idx_np)
        idx_t = torch.from_numpy(idx_np.astype(np.int64))
        render = self._render(indices, pose)
        mx.eval(render, self.bundle.last_commitment_loss)
        result = self.bridge.loss_and_pixel_grad(render, idx_t)
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
        )
        # VQ EMA codebook update from the most-recent forward (van den Oord §3.2).
        self.bundle.ema_update_from_last()
        mx.eval(self.bundle.parameters())
        return {
            "loss": result.loss_value,
            "seg": result.seg_loss_value,
            "pose": result.pose_loss_value,
            "commit": float(self.bundle.last_commitment_loss),
            "d_seg_batch": result.d_seg,
            "grad_clip_would_clip": summary.get(
                "gradient_clip_would_clip_count", 0
            ),
        }

    def exact_d_seg(self) -> float:
        """Mean EXACT live-render d_seg over all pairs (pose-FiLM render)."""
        _require_mlx()
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

    def mean_d_pose(self) -> float:
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

    def configure_stage(self, spec: StageSpec, *, optimizer_schedule: str) -> None:
        """Switch to a PR95 curriculum stage (CurriculumTrainerProtocol).

        Sets the bridge seg-loss family, rebuilds the optimizer config (LR +
        Muon-vs-AdamW per the resolved schedule + per-stage grad-clip/wd), and arms
        the per-stage QAT / C1a / sigma-noise mechanisms. Optimizer + VQ-EMA STATE is
        PRESERVED across stages (PR95 inter-stage transitions resume weights).
        """
        from tac.mlx_pr95_port.curriculum import resolve_use_muon

        self.bridge.set_seg_loss_form(spec.seg_loss_form)
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
        self.mechanisms = StageMechanisms(
            use_qat=spec.use_qat,
            sigma_weight_noise=spec.sigma_weight_noise,
            cat_lambda=spec.cat_lambda,
            cat_sigma=spec.cat_sigma,
        )

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

        d_seg_initial = self.exact_d_seg()
        d_pose_initial = self.mean_d_pose()
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0

        for epoch in range(epochs):
            perm = np.random.permutation(self.n_pairs)
            ep_seg = ep_loss = ep_commit = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np)
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

        d_seg_initial = self.exact_d_seg()
        d_pose_initial = self.mean_d_pose()
        trajectory: list[dict[str, Any]] = []
        best_d_seg = float("inf")
        clip_would_steps = 0
        total_steps = 0

        for epoch in range(cfg.epochs):
            perm = np.random.permutation(self.n_pairs)
            ep_seg = ep_loss = ep_commit = 0.0
            nb = 0
            for start in range(0, self.n_pairs, cfg.batch_size):
                idx_np = perm[start : start + cfg.batch_size]
                row = self.step(idx_np)
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


@torch.no_grad()
def _exact_d_pose(bridge: Any, render_n2chw: Any, idx: torch.Tensor) -> float:
    """Re-measure PoseNet MSE on the live render vs GT pose (the d_pose term)."""
    import torch.nn.functional as F

    np_render = np.asarray(render_n2chw, dtype=np.float32)
    leaf = torch.tensor(np_render, dtype=torch.float32)
    b = leaf.shape[0]
    flat = leaf.reshape(b * 2, 3, leaf.shape[-2], leaf.shape[-1])
    if (int(flat.shape[-2]), int(flat.shape[-1])) != bridge.scorer_hw:
        flat = F.interpolate(
            flat, size=bridge.scorer_hw, mode="bilinear", align_corners=False
        )
    flat = flat.clamp(0.0, 255.0)
    bchw = flat.reshape(b, 2, 3, bridge.scorer_hw[0], bridge.scorer_hw[1])
    bhwc = bchw.permute(0, 1, 3, 4, 2).contiguous()
    posenet_in, _ = bridge.dnet.preprocess_input(bhwc)
    pose_out = bridge.dnet.posenet(posenet_in)
    pose_pred = pose_out["pose"][:, :6]
    target = bridge.pose_targets[idx]
    return float(((pose_pred - target) ** 2).mean().item())


__all__ = [
    "CapstoneTrainConfig",
    "CapstoneTrainer",
]
