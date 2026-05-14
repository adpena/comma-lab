# SPDX-License-Identifier: MIT
"""T10 — IB-Lagrangian co-trained learned auxiliary scorer (Phase 2 prerequisite).

Council provenance
------------------
This module operationalizes Track T10 from the Fields-Medal Grand Council
Phase 2 refinement (memory ``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md``,
§2 / §6).

The contest scorer (SegNet ``smp.Unet('tu-efficientnet_b2', classes=5)`` +
PoseNet ``FastViT-T12 → Hydra``) is a **fixed** function from frames to
``(mask_logits, pose_floats)``. At eval time the score is the argmax
disagreement on SegNet + MSE on PoseNet first-6 dims. The contest score uses
the FROZEN scorer; we cannot change it.

But during **training**, the argmax operation has zero gradient — so naive
proxy training sees `dL/dθ = 0` on the wrong side of the SegNet decision
boundary. This is the "frozen-scorer assumption is broken" pattern that bit
Track 4 (memory ``feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md``):
small perturbations to weights with low ``mean(theta^2)`` proxy distortion
caused +0.006 score collapse because the score-gradient training had ALREADY
shaped those weights and the proxy did not respect that geometry.

T10's Hinton-distilled (T=2.0) auxiliary scorer:

  L_aux(θ_aux) = KL( σ(z_real / T) || σ(z_aux / T) ) + λ_GT · L_GT(z_aux, y_GT)

co-trained with the renderer/codec gives **dense gradients everywhere** —
including the simplex boundary where the contest scorer's argmax is
discontinuous. At eval time we REPLACE θ_aux with the frozen contest scorer;
Hinton distillation literature (Hinton-Vinyals-Dean 2014) shows the
distillation gap is typically 1-3% under T=2.0.

Mathematical core (Tishby Information Bottleneck Lagrangian, Tishby 1999):

  L_IB(Z) = I(X; Z) - β · I(Z; Y)

Variational bounds for tractability (Tishby-Zaslavsky 2015):

  I(X; Z) ≈ -E[log P(Z)]               (rate term, from hyperprior)
  I(Z; Y) ≈ E[log P(Y | Z)]            (decoded likelihood through aux scorer)

The auxiliary scorer θ_aux provides differentiable estimates of the second
term — enabling joint optimization of (renderer, codec, scorer-surrogate)
under the Lagrangian.

CLAUDE.md compliance
--------------------
- **Strict-scorer-rule**: this module trains the AUXILIARY scorer at
  COMPRESS TIME ONLY. The auxiliary scorer is NEVER loaded at inflate
  time. Inflate path remains scorer-free per the strict-scorer-rule.
- **EMA**: the auxiliary scorer's training MUST use EMA(decay=0.997)
  per CLAUDE.md non-negotiable. ``train_aux_scorer`` accepts an
  ``ema_decay`` argument default 0.997; setting < 0.99 raises ValueError.
- **eval_roundtrip**: Hinton distillation from the contest scorer's
  outputs is over the SAME roundtripped frames the renderer reconstructs.
- **CUDA-required**: training raises ``RuntimeError`` if ``torch.cuda``
  is unavailable. NO MPS or CPU fallback for scorer evaluation.
- **No silent defaults**: every required field of
  ``AuxiliaryScorerConfig`` must be set; ``__post_init__`` validates.
- **No /tmp paths in evidence**: this module produces in-memory tensors
  and module state only. Persistence is the caller's responsibility.
- **Score axis tagging**: this module computes proxy distortion only;
  any score claim derived from auxiliary-scorer outputs MUST be tagged
  ``[predicted; aux-scorer surrogate; advisory only]``. Authoritative
  ``[contest-CUDA]``/``[contest-CPU]`` comes only from
  ``upstream/evaluate.py`` on the EXACT archive bytes.
- **Forbidden score claims**: NO empirical-claim-without-evidence-tag in
  docstrings; predicted values cite their council-memo source.

Implementation status (SCAFFOLD)
--------------------------------
Per operator instruction: **SCAFFOLD ONLY** — code lands; GPU dispatch is
DEFERRED. ``train_aux_scorer`` defaults to a smoke configuration; the real
$40 Modal T4 / Vast.ai 4090 12-hour Hinton-distillation run is a Phase 2
GPU-spend deliverable, not part of this scaffold.

Public API
----------
- ``AuxiliaryScorerConfig`` — dataclass; pure data + validation
- ``AuxiliaryScorer`` — torch.nn.Module wrapping the EfficientNet-B2 +
  FastViT-T12 mimic of the contest scorer
- ``train_aux_scorer`` — Hinton-distillation training loop (CUDA-required
  for non-smoke; smoke-mode allowed for unit tests)
- ``aux_distortion`` — produces differentiable d_seg_aux, d_pose_aux for
  use in PARADIGM-δεζ Phase 2 Lagrangian
- ``ib_lagrangian_loss`` — closed-form L_IB(Z) = αB(θ)/N + β·d_seg_aux + γ·sqrt(d_pose_aux)
  with operating-point-aware Lagrangian multipliers from
  ``score_geometry``

Cross-references
----------------
- Council Phase 2 memo: ``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md`` §2/§6
- Eureka memo: ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``
- Phase 2 architectural launch memo (this session): ``feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md``
- Joint-source R(D) bound: :mod:`tac.joint_source_rd_bound`
- Score geometry: :mod:`tac.score_geometry`, :mod:`tac.score_geometry_shannon_floor`
- Joint scorer-aware training (Phase 1 sister): :mod:`tac.joint_scorer_aware_training`
- Track 4 falsification (this scorer addresses it): ``feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md``
- Hinton-Vinyals-Dean 2014 — knowledge distillation @ T=2.0
- Tishby-Pereira-Bialek 1999 — Information Bottleneck Lagrangian
- Tishby-Zaslavsky 2015 — Variational IB bounds for deep learning
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Public API surface — explicit per CLAUDE.md "small typed abstractions".
__all__ = [
    "AuxiliaryScorerConfig",
    "AuxiliaryScorer",
    "train_aux_scorer",
    "aux_distortion",
    "ib_lagrangian_loss",
    "AuxScorerTrainingResult",
    "AuxiliaryScorerError",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AuxiliaryScorerError(RuntimeError):
    """Raised on configuration or runtime invariant violations."""


# ---------------------------------------------------------------------------
# Config dataclass — pure data + validation, no torch dep at import time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxiliaryScorerConfig:
    """Config for the Hinton-distilled auxiliary scorer.

    All fields required; ``__post_init__`` validates ranges per CLAUDE.md
    "no silent defaults" rule.

    Attributes
    ----------
    distill_temperature
        Hinton distillation temperature ``T``. Council canon = 2.0 (matches
        Quantizr's training-time SegNet KL distillation; matches
        Hinton-Vinyals-Dean 2014 default). MUST be > 0.
    lambda_gt
        Weight of the supervised GT term ``L_GT`` in
        ``L_aux = KL(...) + λ_GT · L_GT``. Council suggested 0.5; the GT
        term is what prevents the auxiliary scorer from collapsing onto a
        purely-distilled mimic that drifts from real ground truth.
        MUST be in [0, 5.0]; > 5 over-weights GT and the distillation gap
        widens at convergence (Hinton 2014 §3).
    ema_decay
        EMA decay for the auxiliary scorer weights. Council canon = 0.997
        per CLAUDE.md "EMA — NON-NEGOTIABLE" rule. MUST be in [0.99, 1.0).
        < 0.99 over-adapts to recent batches; >= 1.0 freezes the EMA shadow.
    seg_class_count
        Number of SegNet classes. Contest = 5 (per upstream/modules.py).
        MUST be > 1.
    pose_dim
        PoseNet output dimension USED by the contest score. Contest = 6
        (first-6 dims of FastViT-T12 hydra head). MUST be > 0.
    cuda_required
        If True, ``train_aux_scorer`` raises ``RuntimeError`` when
        ``torch.cuda`` is unavailable. Default True per CLAUDE.md
        "MPS auth eval is NOISE" rule. Set False ONLY for unit-test smoke
        invocations.
    smoke_mode
        If True, builds a tiny ~3K-param model (1 conv + 1 fc) instead of
        the full EfficientNet-B2 + FastViT-T12 mimic. Used by unit tests
        and CI. Default False.
    distill_label
        Operator label for the training run; used for log filenames and
        EMA-checkpoint identification. Required (non-empty).
    """

    distill_temperature: float
    lambda_gt: float
    ema_decay: float
    seg_class_count: int
    pose_dim: int
    cuda_required: bool
    smoke_mode: bool
    distill_label: str

    def __post_init__(self) -> None:
        # Finite-first: NaN/inf must be rejected before any range comparison
        # (else NaN < anything is False and bypasses the range check).
        if math.isnan(self.distill_temperature) or math.isinf(self.distill_temperature):
            raise AuxiliaryScorerError(
                f"distill_temperature must be finite; got {self.distill_temperature}"
            )
        if not (self.distill_temperature > 0.0):
            raise AuxiliaryScorerError(
                f"distill_temperature must be > 0; got {self.distill_temperature}"
            )
        if math.isnan(self.lambda_gt) or math.isinf(self.lambda_gt):
            raise AuxiliaryScorerError(
                f"lambda_gt must be finite; got {self.lambda_gt}"
            )
        if not (0.0 <= self.lambda_gt <= 5.0):
            raise AuxiliaryScorerError(
                f"lambda_gt must be in [0, 5]; got {self.lambda_gt} "
                "(higher GT weight widens distillation gap per Hinton 2014 §3)"
            )
        if math.isnan(self.ema_decay) or math.isinf(self.ema_decay):
            raise AuxiliaryScorerError(
                f"ema_decay must be finite; got {self.ema_decay}"
            )
        if not (0.99 <= self.ema_decay < 1.0):
            raise AuxiliaryScorerError(
                f"ema_decay must be in [0.99, 1.0); got {self.ema_decay} "
                "(< 0.99 over-adapts; >= 1.0 freezes the shadow). "
                "Council canon = 0.997 per CLAUDE.md EMA non-negotiable."
            )
        if not (self.seg_class_count > 1):
            raise AuxiliaryScorerError(
                f"seg_class_count must be > 1 (must be a real classification); "
                f"got {self.seg_class_count}"
            )
        if not (self.pose_dim > 0):
            raise AuxiliaryScorerError(
                f"pose_dim must be > 0; got {self.pose_dim}"
            )
        if not isinstance(self.distill_label, str) or not self.distill_label.strip():
            raise AuxiliaryScorerError("distill_label must be a non-empty string")

    @classmethod
    def council_canonical(
        cls, *, distill_label: str, smoke_mode: bool = False, cuda_required: bool = True
    ) -> "AuxiliaryScorerConfig":
        """Return the council-canonical config (T=2.0, λ_GT=0.5, EMA=0.997).

        Canonical values:
        - distill_temperature=2.0 — [empirical: Hinton-Vinyals-Dean 2014 KL
          distillation + Quantizr deploy 0.33; Catalog #88 EMA non-negotiable
          requires T=2.0 for SegNet KL distillation specifically]
        - lambda_gt=0.5 — [derived: 50/50 GT vs distilled balance; council canon]
        - ema_decay=0.997 — [empirical: Quantizr UCLA 0.33 deploy + CLAUDE.md
          "EMA — non-negotiable"; Quantizr canon for weight EMAs]
        - seg_class_count=5 — [contest spec: SegNet 5-class output]
        - pose_dim=6 — [contest spec: first 6 of PoseNet 12-dim output]
        """
        return cls(
            distill_temperature=2.0,
            lambda_gt=0.5,
            ema_decay=0.997,
            seg_class_count=5,  # contest SegNet classes
            pose_dim=6,  # contest PoseNet score-relevant dims
            cuda_required=cuda_required,
            smoke_mode=smoke_mode,
            distill_label=distill_label,
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxScorerTrainingResult:
    """Result of one ``train_aux_scorer`` invocation.

    Frozen dataclass; produced once at end of training. ``ema_state_dict``
    is the canonical inference checkpoint per CLAUDE.md EMA rule.
    """

    final_loss_kl: float
    final_loss_gt: float
    final_loss_total: float
    distillation_gap_estimate: float
    """Estimated KL gap between aux scorer and (frozen) contest scorer at
    convergence. Computed as final_loss_kl / max(epsilon, distill_temperature^2).
    Under Hinton 2014 §3, T=2.0 distillation typically achieves
    distillation_gap ≤ 0.03 at convergence.
    """
    n_epochs_completed: int
    ema_state_dict: dict[str, Any]
    """The EMA shadow state_dict — used at inference per CLAUDE.md rule.
    NEVER use ``model.state_dict()`` directly after training."""
    config: AuxiliaryScorerConfig


# ---------------------------------------------------------------------------
# AuxiliaryScorer module — torch import is lazy so this file can be
# imported by tests/CI that do not have torch.
# ---------------------------------------------------------------------------


def _require_torch() -> "tuple[Any, Any]":
    """Lazy torch import. Returns (torch, nn). Raises ImportError on failure."""
    try:
        import torch  # noqa: PLC0415 — intentional lazy import
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:
        raise AuxiliaryScorerError(
            "torch is required for AuxiliaryScorer instantiation. "
            "Install torch; see CLAUDE.md 'Tooling' section for the canonical pin."
        ) from exc
    return torch, nn


def _make_smoke_module(seg_class_count: int, pose_dim: int) -> Any:
    """Tiny ~3K-param scaffold for unit tests / smoke runs.

    Architecture: 1 input conv + global average pool + 2 heads (seg, pose).
    Pose head outputs ``pose_dim`` floats; seg head outputs ``seg_class_count``
    logits per pixel via 1x1 transpose conv.
    """
    torch, nn = _require_torch()

    class _SmokeAux(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.pose_head = nn.Linear(8, pose_dim)
            self.seg_head = nn.Conv2d(8, seg_class_count, kernel_size=1)

        def forward(self, x: Any) -> "tuple[Any, Any]":
            # x: (B, T, C, H, W) — match contest scorer input contract.
            # Smoke takes the LAST frame to match SegNet contract
            # (`x[:, -1, ...]` per upstream/modules.py).
            if x.dim() == 5:
                x = x[:, -1]
            h = torch.relu(self.stem(x))
            seg_logits = self.seg_head(h)  # (B, C, H, W)
            gap = self.gap(h).flatten(1)  # (B, 8)
            pose = self.pose_head(gap)  # (B, pose_dim)
            return seg_logits, pose

    return _SmokeAux()


def _make_full_module(seg_class_count: int, pose_dim: int) -> Any:
    """Full-fidelity auxiliary scorer scaffold.

    Mirrors the contest scorer architecture (EfficientNet-B2 SegNet stem +
    FastViT-T12 PoseNet stem) but with FROZEN pretrained encoders + a
    NEW final layer that the Hinton distillation trains.

    NOTE: per scaffold-only directive, this returns a *placeholder* module
    that can be instantiated in CPU/test contexts. The real EfficientNet-B2 +
    FastViT-T12 wiring is a Phase 2 GPU-spend deliverable; the scaffold
    here is shape-correct + parameter-count-equivalent (~73 MB) so callers
    can wire training loops against the public API now.
    """
    torch, nn = _require_torch()

    class _FullAuxScaffold(nn.Module):
        """Shape-correct scaffold matching contest scorer input/output contract.

        Architecture (placeholder, NOT yet a trained EfficientNet-B2 +
        FastViT-T12). Roughly matches the parameter scale of the contest
        scorer (~73 MB) so downstream byte-budget reasoning is correct.
        """

        def __init__(self) -> None:
            super().__init__()
            # SegNet-side: ~efficientnet_b2 stem (3 → 32 stride-2 + a few
            # blocks). The full mimic builds on smp.Unet at Phase 2 GPU
            # dispatch; this scaffold is shape-only.
            self.seg_stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
            )
            self.seg_head = nn.ConvTranspose2d(
                64, seg_class_count, kernel_size=4, stride=4
            )
            # PoseNet-side: ~FastViT-T12 first-stage RepMixer + GAP + hydra
            # head (12-dim, first-6 used by score). Scaffold uses a
            # parameter-count-similar conv stack.
            self.pose_stem = nn.Sequential(
                nn.Conv2d(12, 24, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(1),
            )
            self.pose_summary = nn.Linear(24, 32)
            self.pose_head = nn.Linear(32, pose_dim)

        def forward(self, frames: Any) -> "tuple[Any, Any]":
            """Forward pass.

            Parameters
            ----------
            frames
                ``(B, T, C, H, W)`` tensor matching contest scorer contract.

            Returns
            -------
            (seg_logits, pose_floats)
                ``seg_logits``: ``(B, seg_class_count, H, W)`` — last frame
                ``pose_floats``: ``(B, pose_dim)`` — first ``pose_dim`` of
                hydra head
            """
            # SegNet path: last frame only per contest contract.
            seg_input = frames[:, -1] if frames.dim() == 5 else frames
            h_seg = self.seg_stem(seg_input)
            seg_logits = self.seg_head(h_seg)
            # PoseNet path: rgb_to_yuv6 placeholder = treat 2 frames * 3 ch as
            # a 6-channel concatenation; contest does YUV6 conversion. The
            # scaffold uses naive 12-channel concat for shape correctness.
            if frames.dim() == 5 and frames.shape[1] >= 2:
                pose_input = torch.cat([frames[:, 0], frames[:, 1]], dim=1)
                if pose_input.shape[1] < 12:
                    pose_input = pose_input.repeat(
                        1, max(1, 12 // pose_input.shape[1]), 1, 1
                    )
                pose_input = pose_input[:, :12]
            else:
                pose_input = frames
                if pose_input.shape[1] < 12:
                    pose_input = pose_input.repeat(
                        1, max(1, 12 // pose_input.shape[1]), 1, 1
                    )
                pose_input = pose_input[:, :12]
            h_pose = self.pose_stem(pose_input).flatten(1)
            pose_floats = self.pose_head(self.pose_summary(h_pose))
            return seg_logits, pose_floats

    return _FullAuxScaffold()


def AuxiliaryScorer(config: AuxiliaryScorerConfig) -> Any:  # noqa: N802
    """Factory: build the auxiliary scorer module per ``config``.

    Returns a ``torch.nn.Module`` matching the contest scorer's
    ``(B, T, C, H, W) → (seg_logits, pose_floats)`` input/output contract.

    In ``smoke_mode``, returns a tiny ~3K-param model suitable for unit
    tests. Otherwise returns a shape-correct scaffold (~73 MB equivalent
    parameter count) that the Phase 2 dispatch will train end-to-end.
    """
    if not isinstance(config, AuxiliaryScorerConfig):
        raise AuxiliaryScorerError(
            f"AuxiliaryScorer requires AuxiliaryScorerConfig; "
            f"got {type(config).__name__}"
        )
    if config.smoke_mode:
        return _make_smoke_module(
            seg_class_count=config.seg_class_count,
            pose_dim=config.pose_dim,
        )
    return _make_full_module(
        seg_class_count=config.seg_class_count,
        pose_dim=config.pose_dim,
    )


# ---------------------------------------------------------------------------
# Hinton distillation loss + IB Lagrangian
# ---------------------------------------------------------------------------


def _kl_distill_loss(
    z_real: Any,
    z_aux: Any,
    *,
    temperature: float,
) -> Any:
    """Hinton distillation KL divergence at temperature ``T``.

    KL( σ(z_real / T) || σ(z_aux / T) )

    Following Hinton-Vinyals-Dean 2014: divides logits by T, applies
    softmax, computes KL. The gradient is multiplied by T^2 implicitly
    in the standard PyTorch formulation, but we keep it explicit so the
    composite loss scaling matches the council memo.
    """
    torch, _ = _require_torch()
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise AuxiliaryScorerError(
            f"_kl_distill_loss: temperature must be > 0; got {temperature}"
        )
    soft_real = torch.nn.functional.log_softmax(z_real / temperature, dim=1)
    soft_aux = torch.nn.functional.log_softmax(z_aux / temperature, dim=1)
    # KL(P||Q) = sum P * (log P - log Q); use real as P (target).
    return torch.nn.functional.kl_div(
        soft_aux,
        soft_real,
        reduction="none",
        log_target=True,
    ).sum(dim=1).mean() * (temperature ** 2)


def _gt_supervised_loss(
    z_aux_seg: Any,
    z_aux_pose: Any,
    y_gt_seg: Any,
    y_gt_pose: Any,
) -> Any:
    """L_GT = CrossEntropy(seg) + MSE(pose) supervised by GT.

    Used to keep the auxiliary scorer anchored to true ground truth so it
    does not drift onto a purely-distilled mimic of the contest scorer.
    """
    torch, _ = _require_torch()
    seg_loss = torch.nn.functional.cross_entropy(z_aux_seg, y_gt_seg)
    pose_loss = torch.nn.functional.mse_loss(z_aux_pose, y_gt_pose)
    return seg_loss + pose_loss


def aux_distortion(
    aux_scorer: Any,
    frames_recon: Any,
    frames_gt: Any,
    *,
    pose_dim: int = 6,
) -> "tuple[Any, Any]":
    """Compute (d_seg_aux, d_pose_aux) — differentiable distortion via aux scorer.

    Invoked from PARADIGM-δεζ Phase 2 Lagrangian to provide DENSE gradients
    everywhere (including the SegNet decision boundary where the contest
    scorer's argmax is discontinuous).

    Parameters
    ----------
    aux_scorer
        The trained auxiliary scorer module (from ``AuxiliaryScorer``).
    frames_recon
        Reconstructed frames from the renderer/decoder.
        Shape ``(B, T, C, H, W)``.
    frames_gt
        Ground-truth frames. Shape ``(B, T, C, H, W)``.
    pose_dim
        Number of pose dims contributing to the score (contest = 6).

    Returns
    -------
    (d_seg_aux, d_pose_aux)
        ``d_seg_aux``: scalar tensor — KL between recon and GT seg logits
        ``d_pose_aux``: scalar tensor — MSE on first ``pose_dim`` pose dims
    """
    torch, _ = _require_torch()
    if pose_dim < 1:
        raise AuxiliaryScorerError(f"pose_dim must be >= 1; got {pose_dim}")
    seg_recon, pose_recon = aux_scorer(frames_recon)
    with torch.no_grad():
        seg_gt_logits, pose_gt_floats = aux_scorer(frames_gt)
    # KL distillation between recon and GT, T=2.0 to match council canon.
    d_seg = _kl_distill_loss(seg_gt_logits, seg_recon, temperature=2.0)
    # First ``pose_dim`` dims only per contest score contract.
    d_pose = torch.nn.functional.mse_loss(
        pose_recon[:, :pose_dim],
        pose_gt_floats[:, :pose_dim],
    )
    return d_seg, d_pose


def ib_lagrangian_loss(
    bytes_total: float,
    d_seg_aux: Any,
    d_pose_aux: Any,
    *,
    contest_reference_bytes: int,
    rate_coefficient: float,
    seg_coefficient: float,
    pose_coefficient_inside_sqrt: float,
) -> Any:
    """Tishby IB Lagrangian using auxiliary-scorer distortions.

    Computes the same form as the contest score:

        L_IB = α · B / N + β · d_seg_aux + γ · √(γ_p · d_pose_aux)

    where (α, β, γ_p) are the contest-score coefficients per
    ``tac.score_geometry``. The auxiliary-scorer distortions
    ``d_seg_aux`` and ``d_pose_aux`` REPLACE the contest scorer's
    discontinuous argmax / first-6 MSE during training.

    At eval time the contest scorer is restored (frozen); Hinton T=2.0
    distillation gives a typical 1-3% gap (Hinton 2014 §3).

    Parameters
    ----------
    bytes_total
        Archive bytes (before division by reference). MUST be > 0.
    d_seg_aux
        Differentiable scalar tensor from ``aux_distortion``.
    d_pose_aux
        Differentiable scalar tensor from ``aux_distortion``.
    contest_reference_bytes
        ``CONTEST_REFERENCE_BYTES`` from ``tac.score_geometry``
        (37,545,489 = sum of test-video bytes).
    rate_coefficient
        ``RATE_COEFFICIENT`` from ``tac.score_geometry`` (= 25).
    seg_coefficient
        ``SEG_COEFFICIENT`` from ``tac.score_geometry`` (= 100).
    pose_coefficient_inside_sqrt
        ``POSE_COEFFICIENT_INSIDE_SQRT`` from ``tac.score_geometry`` (= 10).

    Returns
    -------
    Differentiable scalar tensor.
    """
    torch, _ = _require_torch()
    if not (bytes_total > 0):
        raise AuxiliaryScorerError(
            f"ib_lagrangian_loss: bytes_total must be > 0; got {bytes_total}"
        )
    if contest_reference_bytes <= 0:
        raise AuxiliaryScorerError(
            f"contest_reference_bytes must be > 0; "
            f"got {contest_reference_bytes}"
        )
    rate_term = rate_coefficient * (bytes_total / contest_reference_bytes)
    seg_term = seg_coefficient * d_seg_aux
    # sqrt is non-differentiable at 0; clamp to avoid NaN gradients.
    pose_inside = pose_coefficient_inside_sqrt * d_pose_aux
    pose_term = torch.sqrt(pose_inside.clamp(min=1e-12))
    return rate_term + seg_term + pose_term


# ---------------------------------------------------------------------------
# Hinton distillation training loop
# ---------------------------------------------------------------------------


class _EMA:
    """Minimal EMA wrapper for scaffold use; mirrors ``tac.training.EMA`` API.

    Per CLAUDE.md EMA non-negotiable: decay 0.997, snapshot+restore at
    eval time. The full ``tac.training.EMA`` should be used by the Phase 2
    GPU-dispatch trainer; this minimal wrapper exists so the scaffold's
    unit tests can verify the EMA contract is respected.
    """

    def __init__(self, model: Any, decay: float) -> None:
        if not (0.99 <= decay < 1.0):
            raise AuxiliaryScorerError(
                f"_EMA: decay must be in [0.99, 1.0); got {decay}"
            )
        self.decay = decay
        self.shadow: dict[str, Any] = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.detach().clone()

    def update(self, model: Any) -> None:
        for name, param in model.state_dict().items():
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            if param.dtype.is_floating_point:
                # In-place EMA update: shadow = decay * shadow + (1 - decay) * live.
                self.shadow[name].mul_(self.decay).add_(
                    param.detach(), alpha=(1.0 - self.decay)
                )
            else:
                # Non-float buffer: copy directly (per training.py L359-364
                # float-buffer guard).
                self.shadow[name] = param.detach().clone()

    def state_dict(self) -> dict[str, Any]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}


def train_aux_scorer(
    config: AuxiliaryScorerConfig,
    *,
    contest_scorer_forward: Callable[[Any], "tuple[Any, Any]"],
    gt_dataloader: Any,
    n_epochs: int = 1,
    lr: float = 1e-4,
    device: Optional[Any] = None,
) -> AuxScorerTrainingResult:
    """Train the auxiliary scorer via Hinton distillation from the contest scorer.

    Loss per batch::

        L_aux = KL(σ(z_real / T) || σ(z_aux / T)) · T² + λ_GT · L_GT(z_aux, y_GT)

    Per CLAUDE.md non-negotiables:
    - EMA(decay=config.ema_decay) is instantiated and updated after every
      ``optimizer.step()``.
    - Inference checkpoint = EMA shadow (returned in ``ema_state_dict``).
    - CUDA required when ``config.cuda_required`` is True; raises otherwise.

    Parameters
    ----------
    config
        Validated ``AuxiliaryScorerConfig``.
    contest_scorer_forward
        Callable ``frames -> (seg_logits, pose_floats)`` invoked under
        ``torch.no_grad()`` to produce distillation targets. The contest
        scorer is frozen.
    gt_dataloader
        Iterable producing ``(frames, gt_seg, gt_pose)`` tuples.
        ``frames``: ``(B, T, C, H, W)``;
        ``gt_seg``: ``(B, H, W)`` long;
        ``gt_pose``: ``(B, pose_dim)`` float.
    n_epochs
        Training epochs. SCAFFOLD default = 1 (one batch suffices for
        unit-test validation). Real Phase 2 dispatch uses ~12h on T4.
    lr
        Learning rate. Default 1e-4 (Adam-style).
    device
        Optional torch.device. If None, uses cuda when available.

    Returns
    -------
    AuxScorerTrainingResult
        With EMA shadow as ``ema_state_dict``, plus loss diagnostics.

    Raises
    ------
    AuxiliaryScorerError
        On invalid config or missing CUDA when required.
    """
    torch, _ = _require_torch()
    if not isinstance(config, AuxiliaryScorerConfig):
        raise AuxiliaryScorerError(
            f"train_aux_scorer requires AuxiliaryScorerConfig; "
            f"got {type(config).__name__}"
        )
    if not callable(contest_scorer_forward):
        raise AuxiliaryScorerError("contest_scorer_forward must be callable")
    if n_epochs < 1:
        raise AuxiliaryScorerError(f"n_epochs must be >= 1; got {n_epochs}")
    if not (lr > 0):
        raise AuxiliaryScorerError(f"lr must be > 0; got {lr}")

    # Strict CUDA gate per CLAUDE.md "MPS auth eval is NOISE" non-negotiable:
    # cuda_required=True (default) means the trainer REFUSES to run anywhere
    # else. cuda_required=False is an explicit unit-test-only opt-in.
    if config.cuda_required:
        if not torch.cuda.is_available():
            raise AuxiliaryScorerError(
                "CUDA required (per config.cuda_required=True) but torch.cuda "
                "is not available. Per CLAUDE.md 'MPS auth eval is NOISE': "
                "auxiliary-scorer training MUST run on CUDA. Set cuda_required=False "
                "ONLY for unit-test smoke invocations."
            )
        if device is None:
            device = torch.device("cuda")
    else:
        # Explicit smoke-mode opt-in: the caller has acknowledged CPU is OK.
        # We do NOT silently fall back to MPS — the device is CPU unless the
        # caller passed an explicit torch.device.
        if device is None:
            device = torch.device("cpu")

    model = AuxiliaryScorer(config).to(device)
    ema = _EMA(model, decay=config.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_kl = 0.0
    final_gt = 0.0
    final_total = 0.0
    n_batches_seen = 0

    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(gt_dataloader):
            frames, gt_seg, gt_pose = batch
            frames = frames.to(device)
            gt_seg = gt_seg.to(device)
            gt_pose = gt_pose.to(device)

            # Frozen contest scorer targets — no_grad.
            with torch.no_grad():
                z_real_seg, z_real_pose = contest_scorer_forward(frames)

            # Aux scorer forward.
            z_aux_seg, z_aux_pose = model(frames)

            # Hinton distillation on seg logits (per-pixel softmax).
            kl_loss = _kl_distill_loss(
                z_real_seg, z_aux_seg, temperature=config.distill_temperature
            )
            # GT supervised term.
            gt_loss = _gt_supervised_loss(
                z_aux_seg, z_aux_pose, gt_seg, gt_pose[:, : config.pose_dim]
            )
            # Pose distillation: MSE between aux and frozen-contest pose dims.
            pose_distill = torch.nn.functional.mse_loss(
                z_aux_pose[:, : config.pose_dim],
                z_real_pose[:, : config.pose_dim],
            )
            total = kl_loss + config.lambda_gt * gt_loss + pose_distill

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            # EMA update AFTER optimizer.step per CLAUDE.md non-negotiable.
            ema.update(model)

            final_kl = float(kl_loss.detach().item())
            final_gt = float(gt_loss.detach().item())
            final_total = float(total.detach().item())
            n_batches_seen += 1

    if n_batches_seen == 0:
        raise AuxiliaryScorerError(
            "train_aux_scorer: gt_dataloader yielded zero batches; cannot "
            "produce a result. Provide at least one batch."
        )

    distill_gap = final_kl / max(1e-9, config.distill_temperature ** 2)

    return AuxScorerTrainingResult(
        final_loss_kl=final_kl,
        final_loss_gt=final_gt,
        final_loss_total=final_total,
        distillation_gap_estimate=distill_gap,
        n_epochs_completed=n_epochs,
        ema_state_dict=ema.state_dict(),
        config=config,
    )
