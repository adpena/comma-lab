# SPDX-License-Identifier: MIT
"""Torch/CUDA twin of the V9 CGauge level-set training primitives.

This module deliberately owns *math*, not launch policy.  The cloud launcher feeds it
the typed ``spec_v9_cgauge`` argv and the experiment entry point owns persistence and
telemetry.  NumPy remains the portable reference; Torch is accepted only through the
explicit parity probes below.

The linear/FiLM/activation/render ordering mirrors
``levelset_rgb_forward_numpy`` exactly.  ``round_ste`` and ``contest_r`` implement
the differentiable camera-uint8 round trip used by the MLX trainer.  No compile or
fusion is enabled implicitly: CUDA has to re-measure the fp-reorder wall locally.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CudaLevelSetConfig:
    n_pairs: int
    in_feat: int
    hidden_dim: int = 96
    n_hidden: int = 4
    mod_dim: int = 19
    n_classes: int = 5
    activation: str = "hosc"
    hosc_beta: float = 1.0
    hosc_omega: float = 1.0
    softmax_temp: float = 1.0
    chroma: bool = True
    render_h: int = 384
    render_w: int = 512
    camera_h: int = 874
    camera_w: int = 1164


@dataclass(frozen=True)
class TorchExecutionPolicy:
    """Backend-derived throughput policy; never owns scientific values."""

    device_type: str
    amp_dtype: str | None
    grad_scaler: bool
    tf32: bool
    cudnn_benchmark: bool
    compile_mode: str | None
    cuda_graphs: bool
    execution_label: str


def select_torch_execution_policy(device: Any, *, enable_compile: bool = True) -> TorchExecutionPolicy:
    """Select the fastest safe policy from the actual Torch/device capabilities.

    CUDA measurements remain owed: this function reports capabilities and configures no
    scientific parameter. CPU intentionally returns eager fp32 fallbacks.
    """
    import torch

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return TorchExecutionPolicy(
            device_type=dev.type, amp_dtype=None, grad_scaler=False, tf32=False,
            cudnn_benchmark=False, compile_mode=None, cuda_graphs=False,
            execution_label="eager_fallback",
        )
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return TorchExecutionPolicy(
        device_type="cuda", amp_dtype="bfloat16" if bf16 else "float16",
        grad_scaler=not bf16, tf32=True, cudnn_benchmark=True,
        compile_mode="max-autotune" if enable_compile and hasattr(torch, "compile") else None,
        cuda_graphs=bool(hasattr(torch.cuda, "CUDAGraph")),
        execution_label="megakernel_candidate" if enable_compile else "eager_fallback",
    )


def apply_torch_execution_policy(policy: TorchExecutionPolicy) -> None:
    """Apply backend-only fast-math choices authorized for the training loop."""
    import torch

    if policy.device_type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(policy.tf32)
    torch.backends.cudnn.allow_tf32 = bool(policy.tf32)
    torch.backends.cudnn.benchmark = bool(policy.cudnn_benchmark)
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def autocast_context(policy: TorchExecutionPolicy):
    """Return the selected autocast context without forcing CUDA in CPU tests."""
    import torch

    if policy.amp_dtype is None:
        return nullcontext()
    dtype = torch.bfloat16 if policy.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class TorchPoseCarrier:
    """Canonical EON ground-homography frame0 carrier with trainable table dxi."""

    @staticmethod
    def build(xi_stored: np.ndarray, geom: Any, *, residual_scale: float = 1.0):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        xi_np = np.asarray(xi_stored, np.float32)
        if xi_np.ndim != 2 or xi_np.shape[1] != 6:
            raise ValueError(f"xi_stored must be (P,6), got {xi_np.shape}")

        def skew(v):
            z = torch.zeros_like(v[..., 0])
            return torch.stack((
                z, -v[..., 2], v[..., 1],
                v[..., 2], z, -v[..., 0],
                -v[..., 1], v[..., 0], z,
            ), dim=-1).reshape(*v.shape[:-1], 3, 3)

        def exp_se3(xi):
            rho, omega = xi[..., :3], xi[..., 3:]
            W = skew(omega)
            W2 = W @ W
            theta2 = (omega * omega).sum(-1, keepdim=True)[..., None]
            theta = torch.sqrt(theta2.clamp_min(1e-16))
            small = theta2 < 1e-8
            A = torch.where(
                small, 1.0 - theta2 / 6.0 + theta2.square() / 120.0,
                torch.sin(theta) / theta,
            )
            B = torch.where(
                small, 0.5 - theta2 / 24.0 + theta2.square() / 720.0,
                (1.0 - torch.cos(theta)) / theta2,
            )
            C = torch.where(
                small, 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0,
                (theta - torch.sin(theta)) / (theta2 * theta),
            )
            eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 3, 3)
            R = eye + A * W + B * W2
            V = eye + B * W + C * W2
            t = (V @ rho[..., None]).squeeze(-1)
            return R, t

        class _Carrier(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("xi_stored", torch.from_numpy(xi_np.copy()))
                self.register_buffer("K", torch.as_tensor(geom.K, dtype=torch.float32))
                self.register_buffer("Kinv", torch.as_tensor(geom.Kinv, dtype=torch.float32))
                self.register_buffer("plane_n", torch.as_tensor(geom.n, dtype=torch.float32))
                self.register_buffer("target_grid", torch.as_tensor(geom.grid, dtype=torch.float32))
                self.native_hw = tuple(int(x) for x in geom.native_hw)
                self.plane_d = float(geom.d)
                self.dxi = nn.Parameter(torch.zeros_like(self.xi_stored))
                self.residual_scale = float(residual_scale)

            def xi_effective(self, pair_indices):
                return self.xi_stored[pair_indices] + self.residual_scale * self.dxi[pair_indices]

            def forward(self, source_nhwc, pair_indices):
                xi = self.xi_effective(pair_indices)
                if tuple(source_nhwc.shape[1:3]) != self.native_hw:
                    raise ValueError(
                        f"pose source HW {tuple(source_nhwc.shape[1:3])} != geom {self.native_hw}"
                    )
                R, t = exp_se3(xi)
                tn = t[..., :, None] * self.plane_n[None, None, :]
                H = self.K[None] @ (R - tn / self.plane_d) @ self.Kinv[None]
                src_h = torch.linalg.inv(H) @ self.target_grid[None]
                z = src_h[:, 2]
                z_safe = torch.where(z.abs() < 1e-8, torch.ones_like(z), z)
                u, v = src_h[:, 0] / z_safe, src_h[:, 1] / z_safe
                h, w = self.native_hw
                valid = (z > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                grid = torch.stack((
                    2.0 * u / max(w - 1, 1) - 1.0,
                    2.0 * v / max(h - 1, 1) - 1.0,
                ), dim=-1).reshape(-1, h, w, 2)
                x = source_nhwc.permute(0, 3, 1, 2).contiguous()
                warped = F.grid_sample(
                    x, grid, mode="bilinear", padding_mode="border", align_corners=True
                ).permute(0, 2, 3, 1).contiguous()
                return torch.where(valid.reshape(-1, h, w, 1), warped, source_nhwc)

        return _Carrier()


def structured_sdf_prefit(
    model,
    feats,
    target_phi,
    *,
    steps: int,
    lr: float,
    subsample: int,
    seed: int,
) -> dict[str, float | int]:
    """Actually optimize the SDF-producing trunk before main training.

    Callers must skip this function on resume. Only SDF/trunk parameters are routed;
    texture/palette and any attached pose carrier stay untouched.
    """
    import torch

    params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and (
            name.startswith("in_proj") or name.startswith("hidden")
            or name.startswith("film") or name.startswith("out_sdf") or name == "code"
        )
    ]
    if not params:
        raise ValueError("structured SDF prefit found no trainable SDF parameters")
    opt = torch.optim.AdamW(params, lr=float(lr))
    gen = torch.Generator(device=feats.device).manual_seed(int(seed))
    n = int(feats.shape[-2])
    take = min(max(1, int(subsample)), n)
    if feats.ndim != 2 or target_phi.ndim != 2 or feats.shape[0] != target_phi.shape[0]:
        raise ValueError(
            "structured prefit expects pair-0/shared feats (P,F) and static-core target (P,K); "
            f"got feats={tuple(feats.shape)} target={tuple(target_phi.shape)}"
        )
    target = target_phi
    first = last = float("nan")
    model.train()
    for _ in range(int(steps)):
        idx = torch.randperm(n, generator=gen, device=feats.device)[:take]
        opt.zero_grad(set_to_none=True)
        _rgb, phi = model(feats[idx], torch.zeros(1, dtype=torch.long, device=feats.device))
        loss = (phi[0] - target[idx]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("structured SDF prefit produced a nonfinite loss")
        if first != first:
            first = float(loss.detach())
        loss.backward()
        # MLX authority freezes the per-frame code: this is a shared-trunk prior,
        # not a code[0]-only fit.
        if getattr(model, "code", None) is not None and model.code.grad is not None:
            model.code.grad.zero_()
        opt.step()
        last = float(loss.detach())
    return {"steps": int(steps), "loss_initial": first, "loss_final": last}


@dataclass
class DeterministicPairCursor:
    """Crash-resumable exhaustive pair order for true accum-pairs updates."""

    n_pairs: int
    cursor: int = 0
    accepted_total: int = 0
    attempted_total: int = 0

    def next_indices(self, count: int) -> list[int]:
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be positive")
        out = [int((self.cursor + i) % self.n_pairs) for i in range(int(count))]
        self.cursor = int((self.cursor + int(count)) % self.n_pairs)
        self.attempted_total += len(out)
        return out

    def record_accepted(self, count: int) -> None:
        self.accepted_total += int(count)

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.cursor = int(state.get("cursor", 0)) % self.n_pairs
        self.accepted_total = int(state.get("accepted_total", 0))
        self.attempted_total = int(state.get("attempted_total", 0))


def parameter_groups(model) -> dict[str, list[Any]]:
    """Distinct Muon/Adam/pose/code groups used by clipping and optimizer routing."""
    groups: dict[str, list[Any]] = {"muon": [], "adam": [], "pose": [], "code": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "pose_carrier" in name or name.endswith("dxi"):
            groups["pose"].append(p)
        elif name == "code" or name.endswith(".code"):
            groups["code"].append(p)
        elif p.ndim >= 2 and not any(x in name for x in ("out_sdf", "out_tex", "palette", "in_proj")):
            groups["muon"].append(p)
        else:
            groups["adam"].append(p)
    return groups


def clip_grad_groups(groups: Mapping[str, Iterable[Any]], max_norm: float) -> dict[str, Any]:
    """Clip each functional group independently after gradient normalization."""
    import torch

    norms: dict[str, Any] = {}
    for name, params_it in groups.items():
        params = [p for p in params_it if p.grad is not None]
        if not params:
            norms[str(name)] = None
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm), error_if_nonfinite=True)
        norms[str(name)] = norm.detach()
    return norms


@dataclass
class CudaGraphRecaptureGuard:
    """Guard replay across shape/pointer/control/stage changes, not weight updates.

    Parameters and optimizer buffers are mutated *inside* the captured graph. Ordinary
    steps therefore do not invalidate capture; only control-flow or storage topology does.
    """

    generation: int = 0
    captured_generation: int | None = None
    capture_count: int = 0
    replay_count: int = 0

    def invalidate_control_layout(self) -> None:
        self.generation += 1

    def mark_captured(self) -> None:
        self.captured_generation = self.generation
        self.capture_count += 1

    def may_replay(self) -> bool:
        return self.captured_generation == self.generation

    def mark_replayed(self) -> None:
        if not self.may_replay():
            raise RuntimeError("refusing stale CUDA graph replay; recapture required")
        self.replay_count += 1


def round_ste(x):
    """Round in the forward pass and use the identity VJP."""
    return x + (x.round() - x).detach()


def contest_r(rgb_nhwc, *, output_hw: tuple[int, int] = (384, 512)):
    """Contest R: bicubic to camera, uint8 STE, bilinear to scorer resolution."""
    import torch.nn.functional as F

    if rgb_nhwc.ndim == 3:
        rgb_nhwc = rgb_nhwc.unsqueeze(0)
    x = rgb_nhwc.permute(0, 3, 1, 2).contiguous().float()
    x = F.interpolate(x, size=(874, 1164), mode="bicubic", align_corners=False)
    x = round_ste(x).clamp_(0.0, 255.0)
    x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def unified_tau_loss(phi, labels, tau: float):
    """L_tau = tau logsumexp(phi/tau) - phi_y, the V9 unified seg action."""
    import torch

    tau = max(float(tau), 1e-8)
    y = labels.reshape(-1, 1).long()
    flat = phi.reshape(-1, phi.shape[-1])
    correct = torch.gather(flat, 1, y).squeeze(1)
    return (tau * torch.logsumexp(flat / tau, dim=-1) - correct).mean()


def eikonal_and_length(phi_nhwk):
    """Decision-margin Eikonal and Chan-Vese length twins used by the active arm."""
    import torch

    top2 = torch.topk(phi_nhwk, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    gx = 0.5 * (margin[:, 1:-1, 2:] - margin[:, 1:-1, :-2])
    gy = 0.5 * (margin[:, 2:, 1:-1] - margin[:, :-2, 1:-1])
    grad = torch.sqrt(gx.square() + gy.square() + 1e-12)
    eik = (grad - 1.0).square().mean()
    # Smooth delta around the zero/tie locus.  This is the same geometric
    # quantity as the MLX length term, expressed without a backend-only kernel.
    delta = 1.0 / (np.pi * (1.0 + margin[:, 1:-1, 1:-1].square()))
    length = (delta * grad).mean()
    return eik, length


def chroma_boundary_loss(rgb_nhwc, gt_rgb_nhwc, annulus_mask):
    """Luma-invariant BT.601 chroma match on the frozen GT annulus."""
    import torch

    coeff = torch.tensor((0.299, 0.587, 0.114), device=rgb_nhwc.device, dtype=rgb_nhwc.dtype)
    lum = (rgb_nhwc * coeff).sum(-1, keepdim=True)
    gt_lum = (gt_rgb_nhwc * coeff).sum(-1, keepdim=True)
    err = ((rgb_nhwc - lum) - (gt_rgb_nhwc - gt_lum)).square().sum(-1)
    w = annulus_mask.to(err.dtype)
    return (err * w).sum() / w.sum().clamp_min(1.0)


def realized_signed_margin(seg_logits_nhwc, labels_nhw):
    """GT logit minus the strongest competitor on the realized scorer surface."""
    import torch

    labels = labels_nhw.long()
    gt = seg_logits_nhwc.gather(-1, labels[..., None]).squeeze(-1)
    excluded = torch.nn.functional.one_hot(
        labels, num_classes=seg_logits_nhwc.shape[-1]
    ).to(seg_logits_nhwc.dtype)
    runner = (seg_logits_nhwc + excluded * -1e9).amax(dim=-1)
    return gt - runner


def island_birth_from_signed_torch(signed, weight, margin_target: float, *, form: str = "hinge"):
    """Torch twin of ``island_birth_from_signed_np`` on a frozen GT weight map."""
    import torch.nn.functional as F

    deficit = float(margin_target) - signed
    birth = F.softplus(deficit) if form == "softplus" else deficit.clamp_min(0.0)
    w = weight.to(birth.dtype)
    return (birth * w).sum() / (w.sum() + 1e-8)


def area_constraint_torch(seg_logits_nhwc, labels_nhw, lambdas: Mapping[int, float]):
    """One-sided Chan--Vese realized soft-area counterforce."""
    import torch

    soft = torch.softmax(seg_logits_nhwc, dim=-1)
    total = soft.new_zeros(())
    for cls, lam in sorted(lambdas.items()):
        mass = soft[..., int(cls)].mean()
        target = (labels_nhw == int(cls)).to(soft.dtype).mean()
        total = total + 0.5 * float(lam) * (mass - target).clamp_min(0.0).square()
    return total


def _pool3x3_torch(x, kind: str):
    import torch.nn.functional as F

    shape = x.shape
    flat = x.reshape(-1, 1, shape[-2], shape[-1])
    padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
    if kind == "min":
        out = -F.max_pool2d(-padded, 3, stride=1)
    elif kind == "max":
        out = F.max_pool2d(padded, 3, stride=1)
    elif kind == "mean":
        # Replicate padding is the canonical numpy/MLX edge rule.
        out = F.avg_pool2d(padded, 3, stride=1)
    else:
        raise ValueError(kind)
    return out.reshape(shape)


def _soft_skeleton_torch(x, iters: int):
    opened = _pool3x3_torch(_pool3x3_torch(x, "min"), "max")
    skel = (x - opened).clamp_min(0.0)
    for _ in range(int(iters)):
        x = _pool3x3_torch(x, "min")
        opened = _pool3x3_torch(_pool3x3_torch(x, "min"), "max")
        delta = (x - opened).clamp_min(0.0)
        skel = skel + (delta - skel * delta).clamp_min(0.0)
    return skel


def persistence_topology_loss_torch(
    seg_logits_nhwc, labels_nhw, target_classes: tuple[int, ...], *, iters: int = 5
):
    """Differentiable Torch twin of soft-clDice plus island-recall topology loss."""
    import torch

    probs = torch.softmax(seg_logits_nhwc, dim=-1)
    losses = []
    eps = 1e-6
    for n in range(probs.shape[0]):
        for cls in target_classes:
            gt = (labels_nhw[n] == int(cls)).to(probs.dtype)
            if not bool(gt.any()):
                continue
            pred = probs[n, ..., int(cls)]
            sp = _soft_skeleton_torch(pred, iters)
            sg = _soft_skeleton_torch(gt, iters)
            tprec = ((sp * gt).sum() + eps) / (sp.sum() + eps)
            tsens = ((sg * pred).sum() + eps) / (sg.sum() + eps)
            cldice = 1.0 - (2.0 * tprec * tsens / (tprec + tsens + eps))
            density = gt
            for _ in range(4):
                density = _pool3x3_torch(density, "mean")
            w = gt * (1.0 - density).clamp(0.0, 1.0)
            recall = (w * -torch.log(pred.clamp_min(eps))).sum() / (w.sum() + eps)
            losses.append(cldice + recall)
    return torch.stack(losses).mean() if losses else probs.new_zeros(())


def weight_entropy_rate_term_torch(model, *, sigma: float = 0.2):
    """Deterministic soft-histogram rate term, same int8 grid as the MLX/NumPy law."""
    import torch

    total_bits = next(model.parameters()).new_zeros(())
    for name, param in sorted(model.named_parameters()):
        if name == "B" or name.endswith("_B"):
            continue
        scale = param.detach().abs().max() + 1e-8
        grid = (param * (127.0 / scale)).reshape(-1, 1)
        bins = torch.arange(-127, 128, device=param.device, dtype=param.dtype)[None]
        assign = torch.exp(-0.5 * ((grid - bins) / float(sigma)).square())
        assign = assign / (assign.sum(dim=1, keepdim=True) + 1e-12)
        p = assign.mean(dim=0)
        entropy = -(p * torch.log2(p + 1e-12)).sum()
        total_bits = total_bits + entropy * float(param.numel())
    return total_bits, total_bits / 8.0 / 37_545_489.0 * 25.0


def witness_tie_coordinate_torch(signed, direction, eps: float = 1e-6):
    """Torch twin of the shared phase primitive ``t=Mw[p]/(Mw[p]+Mw[q])``."""
    import torch
    import torch.nn.functional as F

    mw = signed.clamp_min(0.0)
    right = F.pad(mw[:, :, 1:], (0, 1, 0, 0))
    down = F.pad(mw[:, 1:, :], (0, 0, 0, 1))
    partner = torch.where(direction < 0.5, right, down)
    return mw / (mw + partner + float(eps))


def homography_grid_from_xi(xi: np.ndarray, geom, *, device, dtype):
    """Build a stop-gradient Torch sampling grid from the fp64 NumPy geometry oracle."""
    import torch

    from tac.boundary_math.warp_real_luma_frame0 import homography_from_xi_numpy

    h, w = geom.native_hw
    hinv = np.linalg.inv(homography_from_xi_numpy(np.asarray(xi, np.float64), geom))
    src = hinv @ geom.grid
    z = src[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = src[0] / z
        v = src[1] / z
    valid = (
        np.isfinite(u) & np.isfinite(v) & (z > 0)
        & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    )
    u = np.clip(u, 0.0, w - 1).reshape(h, w)
    v = np.clip(v, 0.0, h - 1).reshape(h, w)
    grid = np.stack((2.0 * u / max(w - 1, 1) - 1.0, 2.0 * v / max(h - 1, 1) - 1.0), -1)
    return (
        torch.as_tensor(grid, device=device, dtype=dtype)[None],
        torch.as_tensor(valid.reshape(h, w), device=device, dtype=torch.bool)[None, ..., None],
    )


def warp_field_persist_torch(src_nhwc, grid, valid):
    """Bilinear inverse warp with the canonical off-frame PERSIST fallback."""
    import torch
    import torch.nn.functional as F

    sampled = F.grid_sample(
        src_nhwc.permute(0, 3, 1, 2), grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    ).permute(0, 2, 3, 1)
    return torch.where(valid, sampled, src_nhwc)


class TorchLevelSetWitness:  # factory wrapper keeps torch an optional import at module import
    @staticmethod
    def build(cfg: CudaLevelSetConfig, *, seed: int = 0):
        import torch
        import torch.nn as nn

        torch.manual_seed(int(seed))

        class _Witness(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = cfg
                self.in_proj = nn.Linear(cfg.in_feat, cfg.hidden_dim)
                self.hidden = nn.ModuleList(
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_hidden)
                )
                self.film = nn.Linear(cfg.mod_dim, cfg.n_hidden * 2 * cfg.hidden_dim)
                self.out_sdf = nn.Linear(cfg.hidden_dim, cfg.n_classes)
                self.out_tex = nn.Linear(cfg.hidden_dim, 3)
                self.palette = nn.Parameter(torch.empty(cfg.n_classes, 3))
                self.code = nn.Parameter(torch.zeros(2 * cfg.n_pairs, cfg.mod_dim))
                self.register_buffer("softmax_temp", torch.tensor(float(cfg.softmax_temp)))
                self.register_buffer("hosc_beta", torch.tensor(float(cfg.hosc_beta)))
                self._siren_init()

            def _siren_init(self):
                # SIREN initialization is part of the active V9 configuration.
                with torch.no_grad():
                    b = 1.0 / float(self.in_proj.in_features)
                    self.in_proj.weight.uniform_(-b, b)
                    self.in_proj.bias.zero_()
                    for layer in self.hidden:
                        b = np.sqrt(6.0 / layer.in_features) / max(float(cfg.hosc_omega), 1e-8)
                        layer.weight.uniform_(-b, b)
                        layer.bias.zero_()
                    nn.init.xavier_uniform_(self.film.weight)
                    self.film.bias.zero_()
                    nn.init.xavier_uniform_(self.out_sdf.weight)
                    self.out_sdf.bias.zero_()
                    nn.init.xavier_uniform_(self.out_tex.weight)
                    self.out_tex.bias.zero_()
                    self.palette.uniform_(-0.5, 0.5)

            def _act(self, x):
                if cfg.activation == "hosc":
                    return torch.tanh(self.hosc_beta * torch.sin(float(cfg.hosc_omega) * x))
                if cfg.activation == "wire":
                    return torch.cos(20.0 * x) * torch.exp(-((10.0 * x) ** 2))
                return torch.relu(x)

            def _fields(self, feats, code_indices):
                # feats may be shared (P,F) or frame-batched (N,P,F).
                if feats.ndim == 2:
                    feats = feats.unsqueeze(0).expand(code_indices.numel(), -1, -1)
                h = self._act(self.in_proj(feats))
                film = self.film(self.code[code_indices]).reshape(
                    code_indices.numel(), cfg.n_hidden, 2, cfg.hidden_dim
                )
                for li, layer in enumerate(self.hidden):
                    h = self._act(layer(h) * (1.0 + film[:, li, 0, None, :]) + film[:, li, 1, None, :])
                phi = self.out_sdf(h)
                tex = self.out_tex(h)
                soft = torch.softmax(phi / self.softmax_temp.clamp_min(1e-8), dim=-1)
                rgb = torch.sigmoid(soft @ self.palette + tex) * 255.0
                if not cfg.chroma:
                    luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                    rgb = torch.cat((luma, luma, luma), dim=-1)
                return rgb, phi, tex, soft

            def forward(self, feats, code_indices):
                rgb, phi, _tex, _soft = self._fields(feats, code_indices)
                return rgb, phi

            def lane_band_fields(self, feats, code_indices, lane_cls: int = 1):
                """Return pre-R RGB, SDF-soft margin, and learned lane appearance."""
                rgb, phi, tex, soft = self._fields(feats, code_indices)
                top2 = torch.topk(soft, k=2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]
                lane_rgb = torch.sigmoid(self.palette[int(lane_cls)] + tex) * 255.0
                return rgb, phi, margin, lane_rgb

            def render_through_r(self, feats, code_indices, *, lane_band=None):
                if lane_band is None:
                    rgb, phi = self(feats, code_indices)
                else:
                    rgb, phi, margin, lane_rgb = self.lane_band_fields(feats, code_indices)
                    composed = []
                    for ni, code_idx in enumerate(code_indices.detach().cpu().tolist()):
                        prior = lane_band["priors"][int(code_idx) // 2]
                        cov = torch.as_tensor(
                            prior.coverage, device=rgb.device, dtype=rgb.dtype
                        ).reshape(-1)
                        uncertainty = (
                            (float(lane_band["tau"]) - margin[ni])
                            / max(float(lane_band["eps"]), 1e-6)
                            + 0.5
                        ).clamp(0.0, 1.0).detach()
                        alpha = (
                            cov * float(lane_band["weight"]) * uncertainty
                        )[..., None]
                        composed.append(rgb[ni] * (1.0 - alpha) + lane_rgb[ni] * alpha)
                    rgb = torch.stack(composed, dim=0)
                n = rgb.shape[0]
                rgb = rgb.reshape(n, cfg.render_h, cfg.render_w, 3)
                phi = phi.reshape(n, cfg.render_h, cfg.render_w, cfg.n_classes)
                return contest_r(rgb, output_hw=(cfg.render_h, cfg.render_w)), phi

        return _Witness()


def numpy_parameter_dict(model) -> dict[str, np.ndarray]:
    """Export names in the canonical NumPy/MLX checkpoint convention."""
    out: dict[str, np.ndarray] = {
        "in_proj.weight": model.in_proj.weight.detach().cpu().numpy(),
        "in_proj.bias": model.in_proj.bias.detach().cpu().numpy(),
        "film.weight": model.film.weight.detach().cpu().numpy(),
        "film.bias": model.film.bias.detach().cpu().numpy(),
        "out_sdf.weight": model.out_sdf.weight.detach().cpu().numpy(),
        "out_sdf.bias": model.out_sdf.bias.detach().cpu().numpy(),
        "out_tex.weight": model.out_tex.weight.detach().cpu().numpy(),
        "out_tex.bias": model.out_tex.bias.detach().cpu().numpy(),
        "palette": model.palette.detach().cpu().numpy(),
        "code": model.code.detach().cpu().numpy(),
    }
    for i, layer in enumerate(model.hidden):
        out[f"hidden.{i}.weight"] = layer.weight.detach().cpu().numpy()
        out[f"hidden.{i}.bias"] = layer.bias.detach().cpu().numpy()
    return out


def forward_parity_against_numpy(model, feats: np.ndarray, code_index: int = 0) -> dict[str, Any]:
    """MEASURE Torch eager against the portable NumPy reference on identical weights."""
    import torch

    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

    cfg = model.cfg
    p = numpy_parameter_dict(model)
    with torch.inference_mode():
        rgb_t, phi_t = model(
            torch.as_tensor(feats, device=model.code.device),
            torch.tensor([code_index], device=model.code.device),
        )
    rgb_np, phi_np = levelset_rgb_forward_numpy(
        p, feats, p["code"][code_index], n_hidden=cfg.n_hidden,
        hidden_dim=cfg.hidden_dim, n_classes=cfg.n_classes,
        activation=cfg.activation, softmax_temp=float(model.softmax_temp),
        wire_w0=20.0, wire_s0=10.0, hosc_beta=float(model.hosc_beta),
        hosc_omega=cfg.hosc_omega, chroma=cfg.chroma,
    )
    rt = rgb_t[0].detach().cpu().numpy()
    pt = phi_t[0].detach().cpu().numpy()
    return {
        "rgb_max_abs_delta": float(np.max(np.abs(rt - rgb_np))),
        "phi_max_abs_delta": float(np.max(np.abs(pt - phi_np))),
        "argmax_equal": bool(np.array_equal(pt.argmax(-1), phi_np.argmax(-1))),
        "cosine_phi": float(np.dot(pt.ravel(), phi_np.ravel()) / (
            np.linalg.norm(pt.ravel()) * np.linalg.norm(phi_np.ravel()) + 1e-30)),
    }


def compile_identity_probe(model, feats, code_indices, loss_fn) -> dict[str, Any]:
    """Functional compile-adoption probe with advisory loss/gradient deltas.

    Training-loop bit identity is operator-waived. Adoption is therefore based on
    the unchanged score-relevant functional gate: NumPy-fp32 argmax equality and
    ``cosine_phi >= 0.9997``. Loss/gradient deltas remain visible telemetry and are
    never rewritten as exact.
    """
    import torch

    if not hasattr(torch, "compile"):
        return {"available": False, "adoptable": False, "reason": "torch.compile unavailable"}

    def closure():
        rgb, phi = model(feats, code_indices)
        return loss_fn(rgb, phi)

    def run(fn):
        model.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        return loss.detach().clone(), grads

    eager_loss, eager_grad = run(closure)
    try:
        compiled = torch.compile(closure, mode="max-autotune", fullgraph=False)
        comp_loss, comp_grad = run(compiled)
        compiled_forward = torch.compile(model, mode="max-autotune", fullgraph=False)
        with torch.no_grad():
            comp_rgb, comp_phi = compiled_forward(feats, code_indices)
    except Exception as exc:  # compiler support is substrate-specific
        return {"available": True, "adoptable": False, "error": f"{type(exc).__name__}: {exc}"}
    gdelta = max(
        (
            float((a - b).abs().max())
            for a, b in zip(eager_grad, comp_grad, strict=True)
        ),
        default=0.0,
    )
    ldelta = float((eager_loss - comp_loss).abs())
    if feats.ndim == 2 and code_indices.numel() == 1:
        from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

        p = numpy_parameter_dict(model)
        ci = int(code_indices.reshape(-1)[0])
        rgb_np, phi_np = levelset_rgb_forward_numpy(
            p, feats.detach().float().cpu().numpy(), p["code"][ci],
            n_hidden=model.cfg.n_hidden, hidden_dim=model.cfg.hidden_dim,
            n_classes=model.cfg.n_classes, activation=model.cfg.activation,
            softmax_temp=float(model.softmax_temp), wire_w0=20.0, wire_s0=10.0,
            hosc_beta=float(model.hosc_beta), hosc_omega=model.cfg.hosc_omega,
            chroma=model.cfg.chroma,
        )
        cphi = comp_phi[0].detach().float().cpu().numpy()
        crgb = comp_rgb[0].detach().float().cpu().numpy()
        argmax_equal = bool(np.array_equal(cphi.argmax(-1), phi_np.argmax(-1)))
        cosine_phi = float(np.dot(cphi.ravel(), phi_np.ravel()) / (
            np.linalg.norm(cphi.ravel()) * np.linalg.norm(phi_np.ravel()) + 1e-30
        ))
        rgb_max_abs_delta = float(np.max(np.abs(crgb - rgb_np)))
    else:
        with torch.no_grad():
            eager_rgb, eager_phi = model(feats, code_indices)
        argmax_equal = bool(torch.equal(comp_phi.argmax(-1), eager_phi.argmax(-1)))
        cosine_phi = float(torch.nn.functional.cosine_similarity(
            comp_phi.float().reshape(1, -1), eager_phi.float().reshape(1, -1)
        ))
        rgb_max_abs_delta = float((comp_rgb.float() - eager_rgb.float()).abs().max())
    adoptable = bool(argmax_equal and cosine_phi >= 0.9997)
    return {
        "available": True,
        "loss_max_abs_delta": ldelta,
        "grad_max_abs_delta": gdelta,
        "argmax_equal": argmax_equal,
        "cosine_phi": cosine_phi,
        "rgb_max_abs_delta": rgb_max_abs_delta,
        "adoptable": adoptable,
        "adoption_rule": "argmax_equal && cosine_phi>=0.9997",
        "training_loop_bit_identity_waiver": True,
        "law": "witness_fp_reorder_transform_bit_identity_wall_v1",
    }


__all__ = [
    "CudaGraphRecaptureGuard",
    "CudaLevelSetConfig",
    "DeterministicPairCursor",
    "TorchExecutionPolicy",
    "TorchLevelSetWitness",
    "TorchPoseCarrier",
    "apply_torch_execution_policy",
    "area_constraint_torch",
    "autocast_context",
    "chroma_boundary_loss",
    "clip_grad_groups",
    "compile_identity_probe",
    "contest_r",
    "eikonal_and_length",
    "forward_parity_against_numpy",
    "homography_grid_from_xi",
    "island_birth_from_signed_torch",
    "numpy_parameter_dict",
    "parameter_groups",
    "persistence_topology_loss_torch",
    "realized_signed_margin",
    "round_ste",
    "select_torch_execution_policy",
    "structured_sdf_prefit",
    "unified_tau_loss",
    "warp_field_persist_torch",
    "weight_entropy_rate_term_torch",
    "witness_tie_coordinate_torch",
]
