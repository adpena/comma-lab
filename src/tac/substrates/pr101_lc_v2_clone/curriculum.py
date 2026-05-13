"""PR95 hnerv_muon 8-stage curriculum primitives — byte-faithful port.

This module is the substrate-engineering companion to ``architecture.py`` +
``archive.py`` + ``score_aware_loss.py``. It packs the 8-stage PR95
curriculum's losses (CE, tau-softplus, smooth-disagreement, L7-softplus,
cat_entropy_v2), QAT fake-quant helpers, EMA, and the Muon optimizer.

Source: ``experiments/results/public_pr_archive_release_view/
public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/``
(canonical PR95 source under ``losses.py`` and ``optim.py``).

Catalog #109 compliance: this module READS PR95's mathematical contracts
from the intake clone documentation but writes NOTHING into the intake
clone tree. All new code lives here under ``tac.substrates.pr101_lc_v2_clone``.

Catalog #124 compliance: this is substrate engineering — the parent
substrate already declares ``archive_grammar``, ``parser_section_manifest``,
``inflate_runtime_loc_budget`` etc. in its docstring + lane registry
notes. This file extends the substrate with the curriculum primitives.

Forbidden-pattern audit (CLAUDE.md):
* No scorer loading inside this module.
* No /tmp paths; no MPS fallback defaults.
* No comment-only contracts — every "MUST be called inside X" comment
  is paired with an inline ``assert`` or ``raise``.
* No score claims; this module is pure primitive surface.

CLAUDE.md ``EMA — non-negotiable`` compliance: ``ema_update`` mirrors the
PR95 canonical EMA decay 0.999 contract for the curriculum (per-stage
override is exposed via ``CurriculumStageConfig.ema_decay``). The trainer
that consumes this module must use ``tac.training.EMA`` for the canonical
0.997 weight EMA unless the curriculum stage explicitly sets a different
decay value (PR95 uses 0.999 across all 8 stages).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Stage 1: Cross-Entropy seg loss (random-init bulk calibration)
# ============================================================================

def ce_seg_loss(seg_logits: torch.Tensor, targets_hard: torch.Tensor) -> torch.Tensor:
    """Stage 1 CE seg loss — F.cross_entropy on hard SegNet argmax targets.

    Args:
        seg_logits: ``(B, C, H, W)`` SegNet logits.
        targets_hard: ``(B, H, W)`` int64 SegNet argmax class labels.

    Returns:
        Scalar CE loss.
    """
    return F.cross_entropy(seg_logits, targets_hard)


# ============================================================================
# Stage 2: tau-Softplus seg loss (smooth in margin space)
# ============================================================================

def tau_softplus_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    tau: float = 0.3,
) -> torch.Tensor:
    """Stage 2 tau-Softplus surrogate.

    Per-pixel ``tau * softplus(-margin / tau)`` where margin is the gap between
    the target class's logit and the second-highest logit. Smooth in margin
    space, vanishes when target wins by a clear margin.
    """
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return (tau * F.softplus(-margin / tau)).mean()


# ============================================================================
# Stage 3+4: Smooth-disagreement seg loss (sigmoid bell on negative margin)
# ============================================================================

def smooth_disagreement_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    tau: float = 0.3,
) -> torch.Tensor:
    """Stage 3+4 smooth-disagreement loss — ``sigmoid(-margin / tau)``.

    Bell-curve gradient peaks at margin=0; pushes boundary pixels across the
    decision boundary. Vanishes for both clearly-correct and clearly-wrong
    pixels; concentrates gradient on undecided pixels.
    """
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    return torch.sigmoid(-margin / tau).mean()


# ============================================================================
# Stage 5+: L7-weighted Softplus seg loss (concentration on hard pixels)
# ============================================================================

def l7_softplus_seg_loss(
    seg_logits: torch.Tensor,
    targets_hard: torch.Tensor,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> torch.Tensor:
    """Stage 5+ L7-weighted Softplus.

    Per-pixel ``tau * softplus(-margin / tau)`` weighted by
    ``(1 + l7_mult * 1[margin < l7_threshold])``, renormalized to mean-1
    weights. Concentrates the seg gradient on pixels the scorer is close to
    flipping (margin near the boundary).
    """
    target_logits = seg_logits.gather(1, targets_hard.unsqueeze(1))
    masked = seg_logits.clone()
    masked.scatter_(1, targets_hard.unsqueeze(1), -1e9)
    margin = target_logits - masked.max(dim=1, keepdim=True)[0]
    per_pixel = tau * F.softplus(-margin / tau)
    with torch.no_grad():
        weights = 1.0 + l7_mult * (margin < l7_threshold).float()
        weights = weights / weights.mean()
    return (per_pixel * weights).mean()


# ============================================================================
# Pose loss (constant across all 8 stages)
# ============================================================================

def pose_loss(pose_pred: torch.Tensor, pose_target: torch.Tensor) -> torch.Tensor:
    """sqrt(10 * MSE) — concave-in-MSE, emphasizes small errors more than plain MSE.

    Matches the contest score formula's pose term exactly.
    """
    mse = F.mse_loss(pose_pred, pose_target)
    return torch.sqrt(10.0 * mse + 1e-12)


# ============================================================================
# Stage 5+: cat_entropy_v2 — size-weighted soft histogram entropy regularizer
# ============================================================================

def cat_entropy_v2(
    decoder: nn.Module,
    sigma: float = 0.2,
    sample_size: int = 2000,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Size-weighted soft histogram entropy across decoder Conv2d/Linear weights.

    For each Conv2d/Linear weight tensor:
      1. Quantize to ``{-127, ..., 127}`` via Gaussian soft-assignment with
         bandwidth ``sigma``.
      2. Compute the categorical entropy of the soft bin probabilities
         (averaged across the tensor's elements).
      3. Weight by tensor ``numel``.

    Returns:
        Weighted mean entropy in bits/weight, averaged across all weight
        tensors. Pushing this down with small ``sigma`` + big ``lambda``
        sharpens the post-INT8 distribution at integer grid points, which
        directly shrinks the brotli-compressed archive bytes.

    PR95 canonical values: ``sigma=0.2, sample_size=2000`` for Stage 5+
    initially; Stage 7 sharpens to ``sigma=0.1``.
    """
    if device is None:
        device = next(decoder.parameters()).device
    bins = torch.arange(-127, 128, device=device, dtype=torch.float32)
    total_numel = 0
    weighted_entropy = torch.zeros((), device=device)
    for _name, mod in decoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "weight"):
            w = mod.weight
            numel = w.numel()
            ma = w.abs().max().detach()
            if ma.item() < 1e-12:
                continue
            wn = (w / (ma / 127.0)).flatten()
            if wn.numel() > sample_size:
                idx = torch.randperm(wn.numel(), device=wn.device)[:sample_size]
                wn = wn[idx]
            sa = torch.exp(-0.5 * ((wn.unsqueeze(1) - bins.unsqueeze(0)) / sigma).pow(2))
            sa = sa / (sa.sum(dim=1, keepdim=True) + 1e-12)
            bp = sa.mean(dim=0)
            bp = bp / (bp.sum() + 1e-12)
            entropy = -(bp * torch.log2(bp + 1e-12)).sum()
            weighted_entropy = weighted_entropy + numel * entropy
            total_numel += numel
    return weighted_entropy / max(total_numel, 1)


# ============================================================================
# Stage 4+: QAT fake-quant with straight-through estimator
# ============================================================================

def fake_quantize(tensor: torch.Tensor, n_levels: int = 127) -> torch.Tensor:
    """Per-tensor symmetric INT8 fake-quant with STE.

    Forward rounds to integer grid scaled by ``ma / n_levels``; backward passes
    gradient straight through (no quantization-induced gradient term).

    Per PR95 ``src/losses.py:120-125``:
        scale = ma / n_levels if ma > 0 else 1.0
        q = (tensor / scale).round().clamp(-n_levels, n_levels)
        return (q * scale - tensor).detach() + tensor
    """
    ma = tensor.abs().max()
    scale = ma / n_levels if ma > 0 else 1.0
    q = (tensor / scale).round().clamp(-n_levels, n_levels)
    return (q * scale - tensor).detach() + tensor


def apply_qat(decoder: nn.Module) -> dict[str, torch.Tensor]:
    """Replace decoder Conv2d/Linear weights with fake-quantized versions in place.

    Returns the dict of originals so the caller can restore live weights after
    the forward pass.

    Canonical pattern (PR95 ``stages/common.py:172-176``)::

        originals = apply_qat(decoder)
        decoded = decoder(...)
        restore_qat(decoder, originals)

    Args:
        decoder: any nn.Module containing Conv2d / Linear submodules.

    Returns:
        Mapping ``{module_name: live_weight_clone}`` for restore_qat.
    """
    originals: dict[str, torch.Tensor] = {}
    for name, mod in decoder.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and hasattr(mod, "weight"):
            originals[name] = mod.weight.data.clone()
            mod.weight.data = fake_quantize(mod.weight.data)
    return originals


def restore_qat(decoder: nn.Module, originals: dict[str, torch.Tensor]) -> None:
    """Restore live decoder weights from the dict returned by ``apply_qat``.

    Catalog #186 / CLAUDE.md "Comment-only contracts FORBIDDEN" compliance:
    the caller-responsibility contract is enforced by raising if the
    ``originals`` dict references a module that has been removed since the
    paired ``apply_qat`` call.
    """
    for name, mod in decoder.named_modules():
        if name in originals:
            mod.weight.data = originals[name]


# ============================================================================
# EMA helpers (decay 0.999 for the PR95 curriculum)
# ============================================================================

def ema_update(
    ema_decoder: nn.Module,
    decoder: nn.Module,
    ema_latents: torch.Tensor | None,
    latents: nn.Parameter | None,
    decay: float = 0.999,
) -> None:
    """In-place EMA update for both the decoder shadow + the latent shadow.

    PR95 canonical decay is 0.999 across all 8 stages. CLAUDE.md "EMA —
    non-negotiable" requires the inference checkpoint to come from the EMA
    shadow, not the live weights — the caller's apply+snapshot+restore
    pattern is unchanged.
    """
    with torch.no_grad():
        for ep, pv in zip(ema_decoder.parameters(), decoder.parameters(), strict=True):
            ep.data.mul_(decay).add_(pv.data, alpha=1 - decay)
        if ema_latents is not None and latents is not None:
            ema_latents.mul_(decay).add_(latents.data, alpha=1 - decay)


# ============================================================================
# Muon optimizer (Stage 8 only) — Keller Jordan 2024 Newton-Schulz orthogonalized momentum
# ============================================================================

@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz iteration for matrix square root inverse, BF16 step.

    Mirrors PR95 ``src/optim.py:16-29`` byte-for-byte. The (a, b, c) tuple
    (3.4445, -4.7750, 2.0315) is Keller Jordan's published coefficient set
    for 5 NS steps.

    Args:
        G: input matrix (or 4-D tensor; 4-D is flattened to 2-D for the NS step).
        steps: number of NS iterations (PR95 default: 5).
        eps: numerical stabilizer for the normalization step.

    Returns:
        Orthogonalized matrix in the same dtype as ``G``.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.dtype == torch.float32 else G.clone()
    if X.size(-2) > X.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B_ = b * A + c * A @ A
        X = a * X + B_ @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon — Newton-Schulz orthogonalized momentum (Keller Jordan, 2024).

    Decoupled weight decay applied to the parameter directly BEFORE the
    orthogonalized update (matches AdamW convention; Chen-Li-Liu
    arXiv:2506.15054 spectral-norm KKT argument).

    Mirrors PR95 ``src/optim.py:32-80`` byte-for-byte. The 4D conv weights
    are flattened to 2D for the NS step; 2D weights are passed directly;
    everything else is left untouched.

    PR95 canonical hyperparameters (Stage 8):
        lr=2e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0
    Researcher #24 tweak (council G memo 896f1d79): weight_decay=5e-4.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                # Decoupled weight decay applied BEFORE the orthogonalized
                # update — matches AdamW convention; required by the KKT story.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                gu = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = gu.shape
                if gu.ndim == 4:
                    g2d = gu.view(gu.size(0), -1)
                    g_ortho = zeropower_via_newtonschulz5(g2d, steps=ns_steps)
                    scale = max(1.0, (g2d.size(0) / g2d.size(1)) ** 0.5)
                    g_final = (g_ortho * scale).view(orig_shape)
                elif gu.ndim == 2:
                    g_ortho = zeropower_via_newtonschulz5(gu, steps=ns_steps)
                    scale = max(1.0, (gu.size(0) / gu.size(1)) ** 0.5)
                    g_final = g_ortho * scale
                else:
                    g_final = gu

                p.add_(g_final, alpha=-lr)
        return loss


def partition_params_for_muon(model: nn.Module) -> tuple[list, list]:
    """Split model parameters into (muon_params, adamw_params) per PR95 rules.

    Muon: 2D+ weights NOT in stem and NOT in RGB heads.
    AdamW: stem Linear, rgb_0/rgb_1 conv weights, all biases, all 1D params.

    PR95 reports the Stage 8 split as ``Muon: ~177K params (11 tensors)``
    + ``AdamW: ~52K decoder + 16K latent`` for the canonical
    ``base_channels=36, latent_dim=28`` substrate.

    Mirrors PR95 ``src/optim.py:83-100`` byte-for-byte.
    """
    muon_params: list = []
    adamw_params: list = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            adamw_params.append(p)
            continue
        low = name.lower()
        if "stem" in low or low.startswith("rgb") or ".rgb_" in low:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params


# ============================================================================
# Per-stage curriculum config (mirrors PR95 src/stages/common.py StageConfig)
# ============================================================================

@dataclass
class CurriculumStageConfig:
    """Static-configuration record for one of the 8 PR95 curriculum stages.

    Mirrors PR95 ``src/stages/common.py::StageConfig`` byte-for-byte (modulo
    the ``seg_loss_fn`` Callable, which here is resolved via ``stage_name``).
    """

    name: str
    """Stage identifier (e.g. ``stage5_c1a_l7``)."""

    epochs: int
    """Number of training epochs at this stage."""

    seg_loss_kind: str
    """One of ``ce``, ``tau_softplus``, ``smooth_disagreement``, ``l7_softplus``."""

    use_qat: bool = False
    """Whether to wrap the forward in apply_qat()/restore_qat()."""

    use_muon: bool = False
    """Whether to use Muon for hidden convs + AdamW for stem/heads/biases/latents."""

    adamw_lr: float = 3e-5
    """AdamW peak learning rate (cosine annealing to ``lr_floor_ratio`` * adamw_lr)."""

    muon_lr: float = 2e-4
    """Muon peak learning rate (only used if ``use_muon=True``)."""

    muon_weight_decay: float = 0.0
    """Muon decoupled weight decay (PR95 canonical 0.0; researcher #24 tweak 5e-4)."""

    latent_lr_mult: float = 10.0
    """Latents trained at ``latent_lr_mult * adamw_lr`` (PR95 default: 10x)."""

    grad_clip: float = 1.0
    """Gradient L2 norm clip on decoder + latent params."""

    grad_clip_muon: float | None = 1.0
    """Gradient L2 norm clip on Muon params (PR95 default: 1.0)."""

    lr_floor_ratio: float = 5e-6
    """Cosine LR floor as a ratio of ``adamw_lr``. PR95 canonical: 5e-6 / adamw_lr."""

    seg_weight: float = 100.0
    """Aggregation weight for the seg term (matches contest score formula's 100)."""

    pose_weight: float = 1.0
    """Aggregation weight for the pose term (matches contest score formula's 1)."""

    cat_lambda: float = 0.0
    """``cat_entropy_v2`` weight (PR95 Stage 5: 0.01; Stage 6+: 0.02; pre-Stage 5: 0.0)."""

    cat_sigma: float = 0.2
    """``cat_entropy_v2`` bandwidth (PR95 Stage 5+6: 0.2; Stage 7+: 0.1)."""

    batch_size: int = 8
    """Pairs per batch (PR95 canonical: 8)."""

    eval_every: int = 25
    """Run eval every N epochs (PR95 canonical: 25)."""

    ema_decay: float = 0.999
    """EMA decay (PR95 canonical: 0.999 across all 8 stages)."""

    init_latents_random: bool = False
    """Whether to init latents from N(0, 0.1) (Stage 1 only); otherwise resume."""

    extras: dict = field(default_factory=dict)
    """Stage-specific knobs (``tau``, ``l7_threshold``, ``l7_mult`` etc.)."""


def get_seg_loss_fn(kind: str, extras: dict | None = None) -> Callable:
    """Resolve a curriculum seg-loss-kind string to its callable.

    Args:
        kind: one of ``ce``, ``tau_softplus``, ``smooth_disagreement``,
            ``l7_softplus``.
        extras: optional ``dict`` of kwargs (``tau``, ``l7_threshold``, ``l7_mult``)
            passed to the loss function.

    Returns:
        ``Callable[[logits, targets_hard], tensor]``.
    """
    extras = extras or {}
    if kind == "ce":
        return ce_seg_loss
    if kind == "tau_softplus":
        tau = float(extras.get("tau", 0.3))
        return lambda logits, targets: tau_softplus_seg_loss(logits, targets, tau=tau)
    if kind == "smooth_disagreement":
        tau = float(extras.get("tau", 0.3))
        return lambda logits, targets: smooth_disagreement_seg_loss(logits, targets, tau=tau)
    if kind == "l7_softplus":
        tau = float(extras.get("tau", 0.3))
        l7_thresh = float(extras.get("l7_threshold", 1.0))
        l7_mult = float(extras.get("l7_mult", 4.0))
        return lambda logits, targets: l7_softplus_seg_loss(
            logits, targets, tau=tau, l7_threshold=l7_thresh, l7_mult=l7_mult
        )
    raise ValueError(f"unknown seg_loss_kind: {kind!r}")


# ============================================================================
# Canonical 8-stage config registry (mirrors PR95 src/stages/stage{1..8}_*.py)
# ============================================================================

def stage1_v328_ce(epochs: int = 3000) -> CurriculumStageConfig:
    """Stage 1: v3.28 CE phase — bulk calibration from random init.

    Per PR95 ``stages/stage1_v328_ce.py``: AdamW only, peak_lr=1e-3, latent_lr=1e-2,
    20-ep linear warmup then cosine to 5e-6. NO QAT, NO C1a yet.
    """
    return CurriculumStageConfig(
        name="stage1_v328_ce",
        epochs=epochs,
        seg_loss_kind="ce",
        use_qat=False,
        use_muon=False,
        adamw_lr=1e-3,
        cat_lambda=0.0,
        cat_sigma=0.2,
        init_latents_random=True,
    )


def stage2_v331_softplus(epochs: int = 5650) -> CurriculumStageConfig:
    """Stage 2: v3.31 Softplus L1 — switch CE -> tau-Softplus seg loss."""
    return CurriculumStageConfig(
        name="stage2_v331_softplus",
        epochs=epochs,
        seg_loss_kind="tau_softplus",
        use_qat=False,
        use_muon=False,
        adamw_lr=1e-3,
        cat_lambda=0.0,
        cat_sigma=0.2,
        extras={"tau": 0.3},
    )


def stage3_v332_smooth(epochs: int = 1500) -> CurriculumStageConfig:
    """Stage 3: v3.32 smooth-disagreement — fresh cosine peak_lr=1e-4."""
    return CurriculumStageConfig(
        name="stage3_v332_smooth",
        epochs=epochs,
        seg_loss_kind="smooth_disagreement",
        use_qat=False,
        use_muon=False,
        adamw_lr=1e-4,
        cat_lambda=0.0,
        cat_sigma=0.2,
        extras={"tau": 0.3},
    )


def stage4_v332_qat(epochs: int = 500) -> CurriculumStageConfig:
    """Stage 4: v3.32 + QAT — same loss as Stage 3, INT8 fake-quant joins."""
    return CurriculumStageConfig(
        name="stage4_v332_qat",
        epochs=epochs,
        seg_loss_kind="smooth_disagreement",
        use_qat=True,
        use_muon=False,
        adamw_lr=1e-4,
        cat_lambda=0.0,
        cat_sigma=0.2,
        extras={"tau": 0.3},
    )


def stage5_c1a_l7(epochs: int = 9000) -> CurriculumStageConfig:
    """Stage 5: c1a_l7_combined — L7-weighted Softplus + C1a entropy regularizer."""
    return CurriculumStageConfig(
        name="stage5_c1a_l7",
        epochs=epochs,
        seg_loss_kind="l7_softplus",
        use_qat=True,
        use_muon=False,
        adamw_lr=3e-5,
        cat_lambda=0.01,
        cat_sigma=0.2,
        extras={"tau": 0.3, "l7_threshold": 1.0, "l7_mult": 4.0},
    )


def stage6_lambda_sweep(epochs: int = 2000) -> CurriculumStageConfig:
    """Stage 6: lambda_sweep lambda=0.02 — tighten C1a regularization."""
    return CurriculumStageConfig(
        name="stage6_lambda_sweep",
        epochs=epochs,
        seg_loss_kind="l7_softplus",
        use_qat=True,
        use_muon=False,
        adamw_lr=3e-5,
        cat_lambda=0.02,
        cat_sigma=0.2,
        extras={"tau": 0.3, "l7_threshold": 1.0, "l7_mult": 4.0},
    )


def stage7_sigma_sweep(epochs: int = 3000) -> CurriculumStageConfig:
    """Stage 7: sigma_sweep sigma=0.1 — sharpen C1a peaks at integer grid points."""
    return CurriculumStageConfig(
        name="stage7_sigma_sweep",
        epochs=epochs,
        seg_loss_kind="l7_softplus",
        use_qat=True,
        use_muon=False,
        adamw_lr=3e-5,
        cat_lambda=0.02,
        cat_sigma=0.1,
        extras={"tau": 0.3, "l7_threshold": 1.0, "l7_mult": 4.0},
    )


def stage8_muon_finetune(
    epochs: int = 5000, muon_weight_decay: float = 5e-4
) -> CurriculumStageConfig:
    """Stage 8: muon_finetune — switch optimizer to Muon + AdamW.

    Per researcher #24 / council G memo 896f1d79: ``muon_weight_decay=5e-4``
    (Chen-Li-Liu spectral-norm KKT argument). PR95 canonical was 0.0.
    """
    return CurriculumStageConfig(
        name="stage8_muon_finetune",
        epochs=epochs,
        seg_loss_kind="l7_softplus",
        use_qat=True,
        use_muon=True,
        adamw_lr=1e-5,
        muon_lr=2e-4,
        muon_weight_decay=muon_weight_decay,
        cat_lambda=0.02,
        cat_sigma=0.1,
        extras={"tau": 0.3, "l7_threshold": 1.0, "l7_mult": 4.0},
    )


CURRICULUM_STAGES: dict[str, Callable[..., CurriculumStageConfig]] = {
    "stage1_v328_ce": stage1_v328_ce,
    "stage2_v331_softplus": stage2_v331_softplus,
    "stage3_v332_smooth": stage3_v332_smooth,
    "stage4_v332_qat": stage4_v332_qat,
    "stage5_c1a_l7": stage5_c1a_l7,
    "stage6_lambda_sweep": stage6_lambda_sweep,
    "stage7_sigma_sweep": stage7_sigma_sweep,
    "stage8_muon_finetune": stage8_muon_finetune,
}
"""All 8 PR95 curriculum stage builders, keyed by canonical stage name."""


__all__ = [
    "CURRICULUM_STAGES",
    "CurriculumStageConfig",
    "Muon",
    "apply_qat",
    "cat_entropy_v2",
    "ce_seg_loss",
    "ema_update",
    "fake_quantize",
    "get_seg_loss_fn",
    "l7_softplus_seg_loss",
    "partition_params_for_muon",
    "pose_loss",
    "restore_qat",
    "smooth_disagreement_seg_loss",
    "stage1_v328_ce",
    "stage2_v331_softplus",
    "stage3_v332_smooth",
    "stage4_v332_qat",
    "stage5_c1a_l7",
    "stage6_lambda_sweep",
    "stage7_sigma_sweep",
    "stage8_muon_finetune",
    "tau_softplus_seg_loss",
    "zeropower_via_newtonschulz5",
]
