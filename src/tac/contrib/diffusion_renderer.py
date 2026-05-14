# SPDX-License-Identifier: MIT
"""Diffusion-based decoder for mask-conditioned frame synthesis.

Architecture: DDPM denoising diffusion model conditioned on segmentation masks.

The diffusion teacher produces high-quality semantic image synthesis but is
too slow for contest deployment (50-100 denoising steps per frame). The
solution is a three-stage pipeline:

1. Train a diffusion teacher (DiffusionRenderer) on mask→frame with scorer loss
2. Distill the teacher into a fast CNN student (our existing MaskRenderer)
3. Deploy the student only (single forward pass, small, fast)

Includes three distillation strategies:
  - Direct distillation: student matches teacher's full-chain output
  - Progressive distillation (Salimans & Ho 2022): halve steps iteratively
  - Consistency distillation (Song et al. 2023): map any noisy state to clean

All components are modular and compatible with the tac training infrastructure.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Timestep embedding ──────────────────────────────────────────────────


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Following Vaswani et al. (2017) / Ho et al. (2020).

    Args:
        t: (B,) integer or float timestep tensor
        dim: embedding dimension (must be even)

    Returns:
        (B, dim) embedding tensor
    """
    assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ── SPADE conditioning block ────────────────────────────────────────────


class SPADEBlock(nn.Module):
    """Spatially-Adaptive Denormalization (Park et al. 2019).

    Normalizes features with GroupNorm, then modulates with spatially-varying
    gamma/beta computed from the segmentation mask. This gives the denoiser
    class-aware spatial conditioning at every resolution level.

    Args:
        channels: number of feature channels to normalize
        cond_channels: number of conditioning channels (num_classes one-hot)
        hidden: intermediate channels in the conditioning MLP
    """

    def __init__(self, channels: int, cond_channels: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, channels, 3, padding=1)
        # Init: gamma=1 (via zeros → exp or via ones), beta=0
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply SPADE normalization.

        Args:
            x: (B, C, H, W) feature tensor
            cond: (B, cond_channels, H_mask, W_mask) conditioning tensor

        Returns:
            (B, C, H, W) modulated features
        """
        h = self.norm(x)
        # Resize conditioning to match feature spatial dims
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode="nearest")
        shared = self.shared(cond)
        gamma = 1.0 + self.gamma(shared)  # centered at 1
        beta = self.beta(shared)
        return gamma * h + beta


# ── Residual block with timestep + SPADE conditioning ───────────────────


class DiffusionResBlock(nn.Module):
    """Residual block with timestep embedding injection and SPADE conditioning.

    Timestep is injected via adaptive bias (after first conv).
    Mask conditioning via SPADE (after second norm).

    Args:
        channels: feature channels
        cond_channels: mask conditioning channels
        time_dim: timestep embedding dimension
    """

    def __init__(self, channels: int, cond_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.time_proj = nn.Linear(time_dim, channels)
        self.spade = SPADEBlock(channels, cond_channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        # Zero-init second conv for identity residual at start
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) features
            t_emb: (B, time_dim) timestep embedding
            cond: (B, cond_channels, H, W) mask conditioning

        Returns:
            (B, C, H, W) updated features
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Add timestep bias: (B, C) → (B, C, 1, 1)
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.spade(h, cond)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


# ── Conditional U-Net denoiser ──────────────────────────────────────────


class ConditionalUNet(nn.Module):
    """U-Net denoiser with SPADE mask conditioning and timestep embedding.

    Three-level U-Net (full → half → quarter → half → full) with skip
    connections. Each level has DiffusionResBlocks with SPADE conditioning
    and timestep injection.

    Args:
        in_channels: input image channels (3 for noisy RGB)
        out_channels: output channels (3 for predicted noise)
        cond_channels: conditioning channels (num_classes for one-hot mask)
        channels: base channel width (doubled at each downscale level)
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_channels: int = 5,
        channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_dim = time_dim
        ch = channels

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Level 0: full resolution
        self.enc0_res1 = DiffusionResBlock(ch, cond_channels, time_dim)
        self.enc0_res2 = DiffusionResBlock(ch, cond_channels, time_dim)
        self.down0 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)

        # Level 1: half resolution
        self.enc1_res1 = DiffusionResBlock(ch * 2, cond_channels, time_dim)
        self.enc1_res2 = DiffusionResBlock(ch * 2, cond_channels, time_dim)
        self.down1 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)

        # Bottleneck: quarter resolution
        self.mid_res1 = DiffusionResBlock(ch * 4, cond_channels, time_dim)
        self.mid_res2 = DiffusionResBlock(ch * 4, cond_channels, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.dec1_res1 = DiffusionResBlock(ch * 4, cond_channels, time_dim)  # ch*2 skip + ch*2 up
        self.dec1_proj = nn.Conv2d(ch * 4, ch * 2, 1)  # project concatenated back to ch*2
        self.dec1_res2 = DiffusionResBlock(ch * 2, cond_channels, time_dim)

        self.up0 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.dec0_res1 = DiffusionResBlock(ch * 2, cond_channels, time_dim)  # ch skip + ch up
        self.dec0_proj = nn.Conv2d(ch * 2, ch, 1)  # project concatenated back to ch
        self.dec0_res2 = DiffusionResBlock(ch, cond_channels, time_dim)

        # Output
        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy image, timestep, and mask conditioning.

        Args:
            x: (B, 3, H, W) noisy image
            t: (B,) integer timesteps
            cond: (B, num_classes, H, W) one-hot mask conditioning

        Returns:
            (B, 3, H, W) predicted noise
        """
        # Timestep embedding
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        h = self.stem(x)

        # Level 0
        h = self.enc0_res1(h, t_emb, cond)
        skip0 = self.enc0_res2(h, t_emb, cond)
        h = self.down0(skip0)

        # Level 1
        h = self.enc1_res1(h, t_emb, cond)
        skip1 = self.enc1_res2(h, t_emb, cond)
        h = self.down1(skip1)

        # Bottleneck
        h = self.mid_res1(h, t_emb, cond)
        h = self.mid_res2(h, t_emb, cond)

        # Decoder level 1
        h = self.up1(h)
        if h.shape[2:] != skip1.shape[2:]:
            h = F.interpolate(h, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_res1(h, t_emb, cond)
        h = self.dec1_proj(h)
        h = self.dec1_res2(h, t_emb, cond)

        # Decoder level 0
        h = self.up0(h)
        if h.shape[2:] != skip0.shape[2:]:
            h = F.interpolate(h, size=skip0.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, skip0], dim=1)
        h = self.dec0_res1(h, t_emb, cond)
        h = self.dec0_proj(h)
        h = self.dec0_res2(h, t_emb, cond)

        # Output
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)


# ── Diffusion Renderer (Teacher) ────────────────────────────────────────


class DiffusionRenderer(nn.Module):
    """Mask-conditioned denoising diffusion model for frame synthesis.

    Teacher model: high quality but slow (50-100 denoising steps).
    Used to generate training targets for the fast CNN student.

    Implements DDPM (Ho et al. 2020) with:
      - Linear beta schedule
      - SPADE mask conditioning at every U-Net level
      - Sinusoidal timestep embeddings
      - epsilon-prediction parameterization

    Args:
        num_classes: number of segmentation classes (5 for comma SegNet)
        channels: base U-Net channel width (doubled per level)
        num_timesteps: number of diffusion timesteps (noise schedule length)
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        num_classes: int = 5,
        channels: int = 64,
        num_timesteps: int = 100,
        time_dim: int = 128,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.channels = channels

        # U-Net denoiser with mask conditioning
        self.denoiser = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            cond_channels=num_classes,
            channels=channels,
            time_dim=time_dim,
        )

        # Linear noise schedule (Ho et al. 2020)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        # For posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_mean_coef1",
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def _masks_to_onehot(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert integer masks to one-hot float conditioning tensor.

        Args:
            masks: (B, H, W) long tensor with values in [0, num_classes)

        Returns:
            (B, num_classes, H, W) float tensor
        """
        return F.one_hot(masks, self.num_classes).permute(0, 3, 1, 2).float()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean image at timestep t.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_start: (B, 3, H, W) clean image in [0, 1]
            t: (B,) integer timesteps
            noise: optional pre-sampled noise

        Returns:
            (B, 3, H, W) noisy image
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def training_loss(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDPM training loss (simplified, unweighted).

        L = E[||eps - eps_theta(x_t, t, mask)||^2]

        Args:
            frames: (B, 3, H, W) ground truth frames in [0, 255]
            masks: (B, H, W) integer segmentation masks

        Returns:
            scalar loss
        """
        B = frames.shape[0]
        device = frames.device

        # Normalize to [0, 1]
        x_start = frames / 255.0

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Sample noise and create noisy images
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        cond = self._masks_to_onehot(masks)
        noise_pred = self.denoiser(x_noisy, t, cond)

        # MSE loss on noise prediction
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse diffusion step: denoise x_t → x_{t-1}.

        Args:
            x: (B, 3, H, W) noisy image at timestep t
            t: integer timestep (same for all batch elements)
            cond: (B, num_classes, H, W) one-hot mask conditioning

        Returns:
            (B, 3, H, W) denoised image at timestep t-1
        """
        B = x.shape[0]
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

        # Predict noise
        eps_pred = self.denoiser(x, t_batch, cond)

        # Compute predicted x_0
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        x0_pred = (x - sqrt_one_minus * eps_pred) / sqrt_alpha
        x0_pred = x0_pred.clamp(-1.0, 2.0)  # loose clamp for stability

        # Posterior mean
        mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        masks: torch.Tensor,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate frames from masks via full reverse diffusion chain.

        Args:
            masks: (B, H, W) integer segmentation masks
            num_steps: override number of denoising steps (None = all timesteps)

        Returns:
            (B, 3, H, W) generated frames in [0, 255]
        """
        B, H, W = masks.shape
        device = masks.device
        cond = self._masks_to_onehot(masks)

        if num_steps is None:
            num_steps = self.num_timesteps

        # Start from pure noise
        x = torch.randn(B, 3, H, W, device=device)

        # DDPM posterior formula assumes consecutive timesteps. When
        # num_steps < num_timesteps the gaps between steps are large and
        # the DDPM update is mathematically wrong (incorrect beta/alpha).
        # DDIM handles arbitrary step sizes correctly, so delegate to it.
        if num_steps < self.num_timesteps:
            return self.ddim_sample(masks, num_steps=num_steps, eta=0.0)

        step_indices = torch.arange(self.num_timesteps - 1, -1, -1, device=device)

        for t in step_indices:
            x = self.p_sample(x, t.item(), cond)

        # Convert [0, 1] → [0, 255]
        return x.clamp(0.0, 1.0) * 255.0

    def ddim_sample(
        self,
        masks: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling (Song et al. 2020) — deterministic when eta=0.

        Faster than DDPM sampling with comparable quality. With eta=0,
        the sampling is deterministic (same noise → same output).

        Args:
            masks: (B, H, W) integer segmentation masks
            num_steps: number of denoising steps (can be << num_timesteps)
            eta: stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)

        Returns:
            (B, 3, H, W) generated frames in [0, 255]
        """
        B, H, W = masks.shape
        device = masks.device
        cond = self._masks_to_onehot(masks)

        return self._ddim_sample_impl(masks, num_steps, eta, enable_grad=False)

    def ddim_sample_train(
        self,
        masks: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling WITH gradient flow -- for progressive distillation.

        Same algorithm as ddim_sample but without torch.no_grad(), so
        loss.backward() can propagate through the student's denoising steps.

        Args:
            masks: (B, H, W) integer segmentation masks
            num_steps: number of denoising steps
            eta: stochasticity parameter (0 = deterministic DDIM)

        Returns:
            (B, 3, H, W) generated frames in [0, 255]
        """
        return self._ddim_sample_impl(masks, num_steps, eta, enable_grad=True)

    def _ddim_sample_impl(
        self,
        masks: torch.Tensor,
        num_steps: int,
        eta: float,
        enable_grad: bool,
    ) -> torch.Tensor:
        """Shared DDIM sampling implementation.

        Args:
            masks: (B, H, W) integer segmentation masks
            num_steps: number of denoising steps
            eta: stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)
            enable_grad: if False, runs under torch.no_grad()

        Returns:
            (B, 3, H, W) generated frames in [0, 255]
        """
        B, H, W = masks.shape
        device = masks.device
        cond = self._masks_to_onehot(masks)

        # Subsample timestep sequence
        step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()
        # Deduplicate timesteps that collapse to the same int after rounding,
        # which happens when num_steps >= num_timesteps.  Without this, two
        # consecutive loop iterations would use the same (t, t_prev) pair and
        # produce a no-op denoising step.
        step_indices = step_indices.unique(sorted=True, return_inverse=False).flip(0)

        x = torch.randn(B, 3, H, W, device=device)

        ctx = torch.enable_grad() if enable_grad else torch.no_grad()
        with ctx:
            for i in range(len(step_indices) - 1):
                t = step_indices[i].item()
                t_prev = step_indices[i + 1].item()

                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                eps_pred = self.denoiser(x, t_batch, cond)

                # Predicted x_0
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
                x0_pred = (x - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
                x0_pred = x0_pred.clamp(-1.0, 2.0)

                # DDIM update
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * eps_pred
                x = torch.sqrt(alpha_prev) * x0_pred + dir_xt

                if sigma > 0 and t_prev > 0:
                    x = x + sigma * torch.randn_like(x)

        return x.clamp(0.0, 1.0) * 255.0

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Distillation Trainer ────────────────────────────────────────────────


class DistillationTrainer:
    """Distill diffusion teacher into fast CNN student.

    The student learns to map mask → RGB in a single forward pass
    by matching the teacher's denoised outputs plus optional task loss.

    Three loss components:
      1. L1 match to teacher output (distillation signal)
      2. Perceptual match via SegNet features (optional)
      3. Direct scorer loss for task-specific refinement (optional)

    Args:
        teacher: trained DiffusionRenderer (frozen)
        student: fast CNN (MaskRenderer or any mask→RGB module)
        lr: student learning rate
        teacher_steps: number of denoising steps for teacher inference
        use_ddim: use DDIM (deterministic) instead of DDPM sampling
        task_loss_weight: weight for direct scorer loss (0 = pure distillation)
    """

    def __init__(
        self,
        teacher: DiffusionRenderer,
        student: nn.Module,
        lr: float = 1e-4,
        teacher_steps: int = 50,
        use_ddim: bool = True,
        task_loss_weight: float = 0.0,
    ):
        self.teacher = teacher
        self.student = student
        self.teacher_steps = teacher_steps
        self.use_ddim = use_ddim
        self.task_loss_weight = task_loss_weight

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Student optimizer
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-4)

    def generate_teacher_targets(self, masks: torch.Tensor) -> torch.Tensor:
        """Generate high-quality frames from teacher (slow).

        Args:
            masks: (B, H, W) integer segmentation masks

        Returns:
            (B, 3, H, W) teacher-generated frames in [0, 255]
        """
        with torch.no_grad():
            if self.use_ddim:
                return self.teacher.ddim_sample(masks, num_steps=self.teacher_steps)
            return self.teacher.sample(masks, num_steps=self.teacher_steps)

    def distill_step(
        self,
        masks: torch.Tensor,
        gt_frames: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One distillation training step.

        Args:
            masks: (B, H, W) integer segmentation masks
            gt_frames: (B, 3, H, W) optional ground truth for hybrid loss

        Returns:
            dict with loss components
        """
        # Teacher generates targets (slow, no grad)
        teacher_frames = self.generate_teacher_targets(masks)

        # Student forward (fast, with grad)
        student_frames = self.student(masks)

        # Distillation loss: match teacher output
        distill_loss = F.l1_loss(student_frames, teacher_frames)

        # Optional: match ground truth directly (hybrid training)
        gt_loss = torch.tensor(0.0, device=masks.device)
        if gt_frames is not None:
            gt_loss = F.l1_loss(student_frames, gt_frames) * 0.5

        total_loss = distill_loss + gt_loss

        # Backward + update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {
            "distill_loss": distill_loss.item(),
            "gt_loss": gt_loss.item(),
            "total_loss": total_loss.item(),
        }


# ── Progressive Distillation ────────────────────────────────────────────


class ProgressiveDistiller:
    """Progressive distillation: halve denoising steps iteratively.

    Salimans & Ho (2022): train a student to match a teacher's two-step
    output in a single step, then use the student as the new teacher.
    Repeat: 100→50→25→12→6→3→1 step model.

    The 1-step model is effectively a CNN that maps noise+mask → frame.

    Args:
        base_model: trained DiffusionRenderer to start from
        lr: learning rate for each distillation stage
    """

    def __init__(self, base_model: DiffusionRenderer, lr: float = 1e-4):
        self.base_model = base_model
        self.lr = lr
        self.num_classes = base_model.num_classes

    def _create_student(self, teacher: DiffusionRenderer) -> DiffusionRenderer:
        """Create a student with identical architecture but fresh output layer."""
        student = DiffusionRenderer(
            num_classes=teacher.num_classes,
            channels=teacher.channels,
            num_timesteps=teacher.num_timesteps,
            time_dim=teacher.denoiser.time_dim,
        )
        # Initialize from teacher weights
        student.load_state_dict(teacher.state_dict())
        # Re-initialize output layer for fresh learning
        nn.init.zeros_(student.denoiser.out_conv.weight)
        nn.init.zeros_(student.denoiser.out_conv.bias)
        return student

    def distill_stage(
        self,
        teacher: DiffusionRenderer,
        teacher_steps: int,
        student_steps: int,
        masks_loader,
        num_iters: int = 1000,
    ) -> DiffusionRenderer:
        """One stage of progressive distillation.

        Teacher uses teacher_steps, student uses student_steps (= teacher_steps // 2).
        Student learns to match teacher's output in fewer steps.

        Args:
            teacher: current teacher model
            teacher_steps: steps the teacher uses
            student_steps: steps the student will use (teacher_steps // 2)
            masks_loader: iterable yielding (B, H, W) mask batches
            num_iters: training iterations for this stage

        Returns:
            trained student (becomes teacher for next stage)
        """
        student = self._create_student(teacher)
        device = next(teacher.parameters()).device
        student = student.to(device)

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(student.denoiser.parameters(), lr=self.lr, weight_decay=1e-4)

        student.train()
        for i, masks in enumerate(masks_loader):
            if i >= num_iters:
                break

            masks = masks.to(device)

            # Teacher: generate with teacher_steps (deterministic DDIM)
            with torch.no_grad():
                teacher_output = teacher.ddim_sample(masks, num_steps=teacher_steps, eta=0.0)

            # Student: generate with student_steps (use _train variant for gradient flow)
            student_output = student.ddim_sample_train(masks, num_steps=student_steps, eta=0.0)

            # Match teacher output
            loss = F.mse_loss(student_output / 255.0, teacher_output / 255.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.denoiser.parameters(), 1.0)
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"  [prog_distill] stage {teacher_steps}→{student_steps} "
                    f"iter {i + 1}/{num_iters} loss={loss.item():.6f}"
                )

        return student

    def run(
        self,
        masks_loader,
        schedule: list[int] | None = None,
        iters_per_stage: int = 1000,
    ) -> DiffusionRenderer:
        """Run full progressive distillation pipeline.

        Args:
            masks_loader: iterable yielding mask batches (will be reused per stage)
            schedule: step count schedule, e.g. [100, 50, 25, 12, 6, 3, 1]
            iters_per_stage: training iterations per distillation stage

        Returns:
            final distilled model (few-step or single-step)
        """
        if schedule is None:
            T = self.base_model.num_timesteps
            schedule = []
            s = T
            while s > 1:
                schedule.append(s)
                s = max(1, s // 2)
            schedule.append(1)

        print(f"[prog_distill] Schedule: {schedule}")
        teacher = self.base_model

        for i in range(len(schedule) - 1):
            teacher_steps = schedule[i]
            student_steps = schedule[i + 1]
            print(f"[prog_distill] Stage {i + 1}/{len(schedule) - 1}: {teacher_steps}→{student_steps} steps")
            teacher = self.distill_stage(
                teacher,
                teacher_steps,
                student_steps,
                masks_loader,
                num_iters=iters_per_stage,
            )

        return teacher


# ── Consistency Model ───────────────────────────────────────────────────


class ConsistencyModel(nn.Module):
    """Consistency model (Song et al. 2023) for single-step generation.

    Maps any point on the diffusion trajectory directly to the clean output.
    Key property: f(x_t, t) = f(x_{t'}, t') for any t, t' on the same
    trajectory. This is enforced by the consistency loss.

    Can be trained by:
      1. Consistency distillation (CD): from a pre-trained diffusion teacher
      2. Consistency training (CT): from scratch (not implemented here)

    The network architecture is the same ConditionalUNet but with a skip
    connection that ensures f(x_0, 0) = x_0 (boundary condition).

    Args:
        num_classes: segmentation classes
        channels: base U-Net channel width
        num_timesteps: number of discretization steps
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        num_classes: int = 5,
        channels: int = 64,
        num_timesteps: int = 100,
        time_dim: int = 128,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.channels = channels

        self.net = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            cond_channels=num_classes,
            channels=channels,
            time_dim=time_dim,
        )

        # Noise schedule (shared with teacher)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _masks_to_onehot(self, masks: torch.Tensor) -> torch.Tensor:
        return F.one_hot(masks, self.num_classes).permute(0, 3, 1, 2).float()

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Map noisy input at timestep t to clean output prediction.

        Enforces boundary condition: at t=0, output = input (skip connection).

        f_theta(x_t, t) = c_skip(t) * x_t + c_out(t) * F_theta(x_t, t)

        where c_skip(0) = 1, c_out(0) = 0.

        Args:
            x_t: (B, 3, H, W) noisy image
            t: (B,) timesteps
            cond: (B, num_classes, H, W) one-hot mask conditioning

        Returns:
            (B, 3, H, W) predicted clean image
        """
        # Skip connection coefficients that enforce boundary condition
        # c_skip(t) approaches 1 as t→0, c_out(t) approaches 0 as t→0
        sigma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        c_skip = 1.0 / (1.0 + sigma**2)
        c_out = sigma / (1.0 + sigma**2).sqrt()

        # Network prediction
        F_pred = self.net(x_t, t, cond)

        return c_skip * x_t + c_out * F_pred

    @torch.no_grad()
    def sample(
        self,
        masks: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Generate frames in num_steps (ideally 1-2).

        For single-step: sample noise, apply consistency function once.
        For multi-step: iteratively denoise using the consistency property.

        Args:
            masks: (B, H, W) integer segmentation masks
            num_steps: number of generation steps (1 for single-step)

        Returns:
            (B, 3, H, W) generated frames in [0, 255]
        """
        B, H, W = masks.shape
        device = masks.device
        cond = self._masks_to_onehot(masks)

        x = torch.randn(B, 3, H, W, device=device)

        if num_steps == 1:
            # Single step: directly map noise to clean image
            t = torch.full((B,), self.num_timesteps - 1, device=device, dtype=torch.long)
            x = self.forward(x, t, cond)
        else:
            # Multi-step: iteratively apply consistency function
            step_indices = torch.linspace(self.num_timesteps - 1, 0, num_steps + 1, device=device).long()

            for i in range(num_steps):
                t_now = step_indices[i]
                t_next = step_indices[i + 1]
                t_batch = torch.full((B,), t_now.item(), device=device, dtype=torch.long)

                # Predict clean image
                x0_pred = self.forward(x, t_batch, cond)

                if t_next > 0:
                    # Re-noise to t_next for the next iteration
                    noise = torch.randn_like(x)
                    sqrt_alpha = self.sqrt_alphas_cumprod[t_next]
                    sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t_next]
                    x = sqrt_alpha * x0_pred + sqrt_one_minus * noise
                else:
                    x = x0_pred

        return x.clamp(0.0, 1.0) * 255.0

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConsistencyDistiller:
    """Distill a diffusion teacher into a consistency model.

    Song et al. (2023) consistency distillation: enforce that the model
    maps adjacent points on the same ODE trajectory to the same output.

    Loss: ||f_theta(x_{t_{n+1}}, t_{n+1}) - f_theta_ema(x_{t_n}, t_n)||^2

    where x_{t_n} is obtained by one ODE step from x_{t_{n+1}} using the
    teacher's predicted noise.

    Args:
        teacher: pre-trained DiffusionRenderer
        student: ConsistencyModel to train
        lr: learning rate
        ema_decay: EMA decay for the target network
    """

    def __init__(
        self,
        teacher: DiffusionRenderer,
        student: ConsistencyModel,
        lr: float = 1e-4,
        ema_decay: float = 0.999,
    ):
        self.teacher = teacher
        self.student = student
        self.ema_decay = ema_decay

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # EMA copy of student (target network)
        self.target = ConsistencyModel(
            num_classes=student.num_classes,
            channels=student.channels,
            num_timesteps=student.num_timesteps,
            time_dim=student.net.time_dim,
        )
        self.target.load_state_dict(student.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=1e-4)

    @torch.no_grad()
    def _update_ema(self):
        """Update target network with EMA of student weights."""
        for p_target, p_student in zip(self.target.parameters(), self.student.parameters()):
            p_target.data.lerp_(p_student.data, 1.0 - self.ema_decay)

    def distill_step(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, float]:
        """One consistency distillation step.

        Args:
            frames: (B, 3, H, W) ground truth frames in [0, 255]
            masks: (B, H, W) integer segmentation masks

        Returns:
            dict with loss value
        """
        B = frames.shape[0]
        device = frames.device
        x_start = frames / 255.0
        cond = self.student._masks_to_onehot(masks)

        # Sample adjacent timesteps t_{n+1} and t_n
        n = torch.randint(1, self.student.num_timesteps, (B,), device=device)
        n_prev = n - 1

        # Forward diffuse to t_{n+1}
        noise = torch.randn_like(x_start)
        sqrt_alpha_n = self.student.sqrt_alphas_cumprod[n].view(-1, 1, 1, 1)
        sqrt_one_minus_n = self.student.sqrt_one_minus_alphas_cumprod[n].view(-1, 1, 1, 1)
        x_n = sqrt_alpha_n * x_start + sqrt_one_minus_n * noise

        # Use teacher to estimate x at t_n (one DDIM step backward)
        with torch.no_grad():
            t_batch = n
            eps_pred = self.teacher.denoiser(x_n, t_batch, cond)
            alpha_n = self.teacher.alphas_cumprod[n].view(-1, 1, 1, 1)
            alpha_prev = self.teacher.alphas_cumprod[n_prev].view(-1, 1, 1, 1)
            x0_pred = (x_n - torch.sqrt(1 - alpha_n) * eps_pred) / torch.sqrt(alpha_n)
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            # DDIM step to t_n
            x_n_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps_pred

        # Student: predict clean image from x_{t_{n+1}}
        student_pred = self.student(x_n, n, cond)

        # Target: predict clean image from x_{t_n} (using EMA weights)
        with torch.no_grad():
            target_pred = self.target(x_n_prev, n_prev, cond)

        # Consistency loss: both should predict the same clean image
        loss = F.mse_loss(student_pred, target_pred)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self._update_ema()

        return {"consistency_loss": loss.item()}


# ── Factory functions ───────────────────────────────────────────────────


def build_diffusion_teacher(
    num_classes: int = 5,
    channels: int = 64,
    num_timesteps: int = 100,
    time_dim: int = 128,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionRenderer:
    """Build a diffusion teacher model.

    Default: ~2.5M params with channels=64, 3-level U-Net.

    Args:
        num_classes: segmentation classes (5 for comma)
        channels: base channel width
        num_timesteps: noise schedule length
        time_dim: timestep embedding dimension
        beta_start: noise schedule start (lower bound)
        beta_end: noise schedule end (upper bound)

    Returns:
        DiffusionRenderer ready for training
    """
    model = DiffusionRenderer(
        num_classes=num_classes,
        channels=channels,
        num_timesteps=num_timesteps,
        time_dim=time_dim,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    total = model.param_count()
    print(
        f"[diffusion] Built DiffusionRenderer: {total:,} params "
        f"(channels={channels}, T={num_timesteps}, time_dim={time_dim})"
    )
    return model


def build_consistency_model(
    num_classes: int = 5,
    channels: int = 64,
    num_timesteps: int = 100,
    time_dim: int = 128,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> ConsistencyModel:
    """Build a consistency model for single-step generation.

    Same architecture as the diffusion teacher, but with skip-connection
    parameterization for consistency property.

    Args:
        num_classes: segmentation classes (5 for comma)
        channels: base channel width
        num_timesteps: discretization steps
        time_dim: timestep embedding dimension
        beta_start: noise schedule start (lower bound)
        beta_end: noise schedule end (upper bound)

    Returns:
        ConsistencyModel ready for distillation or training
    """
    model = ConsistencyModel(
        num_classes=num_classes,
        channels=channels,
        num_timesteps=num_timesteps,
        time_dim=time_dim,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    total = model.param_count()
    print(f"[consistency] Built ConsistencyModel: {total:,} params (channels={channels}, T={num_timesteps})")
    return model
