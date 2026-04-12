"""Finance & HFT optimization algorithms for scorer-optimal pixel generation.

Cross-disciplinary transfer from quantitative finance and high-frequency trading
to the video compression scoring problem:

    S = 100*seg_distortion + sqrt(10*pose_distortion) + 25*rate

We optimize 384x512x3 pixel values against frozen scorer networks (PoseNet, SegNet).

Finance insight: this is an EXECUTION problem, not just optimization.
- The loss landscape has "liquidity" (flat = liquid, sharp = illiquid)
- Gradient noise is heteroskedastic (GARCH, not iid)
- The 3-component score is a PORTFOLIO with correlated assets
- Optimal execution speed depends on market impact (overshooting)

Algorithms:
    1. Almgren-Chriss Optimal Execution (trajectory scheduling)
    2. Kelly Criterion (gradient step sizing)
    3. Black-Scholes Implied Volatility (per-pixel sensitivity map)
    4. Markowitz Mean-Variance Portfolio (gradient budget allocation)
    5. Pairs Trading / Statistical Arbitrage (correlated pixel blocks)
    6. GARCH Volatility (adaptive exploration noise)
    7. Order Book Dynamics (priority queue pixel optimization)
    8. Avellaneda-Stoikov Market Making (perturbation range)
    9. Momentum / Mean Reversion Hybrid (adaptive signal following)
   10. Risk Parity (equal marginal contribution across score components)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

__all__ = [
    "FinanceOptimizer",
    "AlmgrenChrissOptimizer",
    "KellyCriterionOptimizer",
    "ImpliedVolatilityOptimizer",
    "MarkowitzOptimizer",
    "PairsTradingOptimizer",
    "GARCHVolatilityOptimizer",
    "OrderBookOptimizer",
    "AvellanedaStoikovOptimizer",
    "MomentumMeanReversionOptimizer",
    "RiskParityOptimizer",
    "FinanceEnsembleOptimizer",
    "smoke_test_all",
    "yousfi_contrarian_picks",
    "OPTIMIZER_REGISTRY",
    "get_optimizer",
]


# ── Base class ──────────────────────────────────────────────────────────


class FinanceOptimizer:
    """Base class for finance/HFT-inspired optimization algorithms.

    Each optimizer wraps around a standard gradient-based loop and modifies
    the step direction, step size, or exploration strategy using insights
    from quantitative finance and high-frequency trading.

    Subclasses must implement:
        - step(pixels, grad, iteration) -> pixels
        - state_dict() -> dict  (for checkpointing)
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.iteration = 0

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Apply one optimization step.

        Args:
            pixels: (H, W, 3) or (N, H, W, 3) current pixel values in [0, 255].
            grad: same shape as pixels, gradient of loss w.r.t. pixels.
            iteration: current iteration number.

        Returns:
            Updated pixels, clamped to [0, 255].
        """
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing."""
        return {"iteration": self.iteration, "config": self.config}

    @staticmethod
    def _clamp_pixels(pixels: torch.Tensor) -> torch.Tensor:
        return pixels.clamp(0.0, 255.0)


# ── 1. Almgren-Chriss Optimal Execution ────────────────────────────────


class AlmgrenChrissOptimizer(FinanceOptimizer):
    """Optimal execution trajectory through loss space.

    In finance: minimize market impact when executing a large trade over T periods.
    Here: minimize "scorer impact" (overshooting sharp minima) when optimizing
    pixels over N iterations.

    The optimal execution schedule from Almgren-Chriss (2000) gives us
    time-varying step sizes that are large early (when far from optimum,
    landscape is smooth) and small late (when near optimum, landscape is sharp).

    The schedule is:
        n_k = total_budget * sinh(kappa * (T-k)/T) / sum_j sinh(kappa * (T-j)/T)

    where kappa = sqrt(lambda * sigma^2 / eta) controls the urgency/caution tradeoff.
    - High kappa (high risk aversion) -> front-loaded execution (aggressive early)
    - Low kappa -> uniform execution (steady pace)

    Config:
        total_steps (int): total optimization iterations T. Default 1000.
        risk_aversion (float): lambda, penalizes variance of trajectory. Default 1e-3.
        volatility (float): sigma, estimated loss landscape volatility. Default 10.0.
        temporary_impact (float): eta, how much each step disturbs the landscape. Default 1.0.
        base_lr (float): base learning rate before schedule scaling. Default 1.0.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.total_steps = config.get("total_steps", 1000)
        self.risk_aversion = config.get("risk_aversion", 1e-3)
        self.volatility = config.get("volatility", 10.0)
        self.temporary_impact = config.get("temporary_impact", 1.0)
        self.base_lr = config.get("base_lr", 1.0)

        # Precompute kappa and the optimal schedule
        self.kappa = math.sqrt(
            self.risk_aversion * self.volatility**2 / (self.temporary_impact + 1e-12)
        )
        # Precompute schedule weights: sinh(kappa * (T-k)/T) for k=0..T-1
        schedule = []
        for k in range(self.total_steps):
            frac = (self.total_steps - k) / self.total_steps
            schedule.append(math.sinh(self.kappa * frac + 1e-8))
        total = sum(schedule) + 1e-12
        self.schedule = [s / total for s in schedule]

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        idx = min(iteration, self.total_steps - 1)
        lr = self.base_lr * self.schedule[idx] * self.total_steps
        pixels = pixels - lr * grad
        return self._clamp_pixels(pixels)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["kappa"] = self.kappa
        return d


# ── 2. Kelly Criterion for Step Sizing ─────────────────────────────────


class KellyCriterionOptimizer(FinanceOptimizer):
    """Optimal step sizing based on gradient reliability.

    Kelly (1956): optimal bet fraction f* = (p*b - q) / b
    - p = probability the gradient direction is correct
    - q = 1 - p
    - b = expected improvement ratio (payoff odds)

    We estimate p from the signal-to-noise ratio of the gradient:
        SNR = |mean(grad)| / (std(grad) + eps)
        p = sigmoid(snr_scale * SNR)  (maps SNR to probability)

    The payoff b is estimated from the ratio of recent loss improvements
    to gradient norms. When gradients are reliable (high SNR), Kelly says
    take large steps. When noisy, take small steps.

    Config:
        base_lr (float): maximum learning rate. Default 1.0.
        snr_scale (float): scaling for SNR -> probability mapping. Default 2.0.
        min_fraction (float): minimum Kelly fraction (floor). Default 0.01.
        momentum (float): EMA for SNR tracking. Default 0.9.
        window (int): number of recent gradients for SNR computation. Default 10.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.snr_scale = config.get("snr_scale", 2.0)
        self.min_fraction = config.get("min_fraction", 0.01)
        self.momentum = config.get("momentum", 0.9)

        # Running EMA of gradient mean and variance
        self.grad_mean_ema: torch.Tensor | None = None
        self.grad_var_ema: torch.Tensor | None = None

    def _update_snr(self, grad: torch.Tensor) -> torch.Tensor:
        """Update running SNR estimate, return per-pixel SNR."""
        # Compute per-channel statistics over spatial dims
        flat = grad.reshape(-1)
        mean = grad.mean()
        var = grad.var() + 1e-8

        if self.grad_mean_ema is None:
            self.grad_mean_ema = mean.detach()
            self.grad_var_ema = var.detach()
        else:
            m = self.momentum
            self.grad_mean_ema = m * self.grad_mean_ema + (1 - m) * mean.detach()
            self.grad_var_ema = m * self.grad_var_ema + (1 - m) * var.detach()

        snr = self.grad_mean_ema.abs() / (self.grad_var_ema.sqrt() + 1e-8)
        return snr

    def _kelly_fraction(self, snr: torch.Tensor) -> torch.Tensor:
        """Map SNR to Kelly fraction via sigmoid."""
        p = torch.sigmoid(self.snr_scale * snr)
        # Assume b=1 (symmetric payoff): f* = 2p - 1, clipped to [min_fraction, 1]
        f_star = (2 * p - 1).clamp(min=self.min_fraction, max=1.0)
        return f_star

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        snr = self._update_snr(grad)
        fraction = self._kelly_fraction(snr)
        lr = self.base_lr * fraction.item()
        pixels = pixels - lr * grad
        return self._clamp_pixels(pixels)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        if self.grad_mean_ema is not None:
            d["grad_mean_ema"] = self.grad_mean_ema.item()
            d["grad_var_ema"] = self.grad_var_ema.item()
        return d


# ── 3. Black-Scholes Implied Volatility ────────────────────────────────


class ImpliedVolatilityOptimizer(FinanceOptimizer):
    """Per-pixel implied volatility for spatially-adaptive optimization.

    Black-Scholes option pricing uses implied volatility (IV) backed out from
    market prices. High IV = uncertain, low IV = stable.

    We compute per-pixel "implied volatility" from gradient history:
        sigma_implied(i,j) = std(grad_history(i,j)) / (|mean(grad_history(i,j))| + eps)

    This creates a spatial heat map:
    - High IV pixels: scorer is sensitive here, prioritize optimization (explore)
    - Low IV pixels: scorer is stable here, preserve current values (exploit)

    The optimizer uses IV to modulate per-pixel learning rate:
        lr(i,j) = base_lr * (iv(i,j) / mean_iv)^power

    Config:
        base_lr (float): base learning rate. Default 1.0.
        power (float): exponent for IV-based LR scaling. Default 0.5.
        ema_decay (float): EMA decay for gradient statistics. Default 0.95.
        min_iv (float): floor for implied volatility. Default 0.01.
        exploration_noise (float): noise magnitude for high-IV regions. Default 0.0.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.power = config.get("power", 0.5)
        self.ema_decay = config.get("ema_decay", 0.95)
        self.min_iv = config.get("min_iv", 0.01)
        self.exploration_noise = config.get("exploration_noise", 0.0)

        # Per-pixel running stats
        self._grad_mean: torch.Tensor | None = None
        self._grad_sq_mean: torch.Tensor | None = None

    def _update_iv(self, grad: torch.Tensor) -> torch.Tensor:
        """Update implied volatility map from gradient."""
        d = self.ema_decay
        if self._grad_mean is None:
            self._grad_mean = grad.detach().clone()
            self._grad_sq_mean = (grad**2).detach().clone()
        else:
            self._grad_mean = d * self._grad_mean + (1 - d) * grad.detach()
            self._grad_sq_mean = d * self._grad_sq_mean + (1 - d) * (grad**2).detach()

        variance = (self._grad_sq_mean - self._grad_mean**2).clamp(min=1e-12)
        std = variance.sqrt()
        mean_abs = self._grad_mean.abs() + 1e-8
        iv = (std / mean_abs).clamp(min=self.min_iv)
        return iv

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        iv = self._update_iv(grad)

        # Normalize IV so mean = 1
        mean_iv = iv.mean() + 1e-8
        iv_normalized = iv / mean_iv

        # Per-pixel LR: high IV -> higher LR
        lr_map = self.base_lr * iv_normalized.pow(self.power)
        pixels = pixels - lr_map * grad

        # Optional exploration noise in high-IV regions
        if self.exploration_noise > 0:
            noise = torch.randn_like(pixels) * self.exploration_noise * iv_normalized
            pixels = pixels + noise

        return self._clamp_pixels(pixels)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["has_stats"] = self._grad_mean is not None
        return d


# ── 4. Markowitz Mean-Variance Portfolio ───────────────────────────────


class MarkowitzOptimizer(FinanceOptimizer):
    """Gradient budget allocation across pixel blocks via mean-variance optimization.

    Markowitz (1952): allocate capital across assets to maximize E[return]/Var[return].

    Here each 8x8 pixel block is an "asset" with:
    - Expected return: predicted score improvement from optimizing this block
    - Risk: variance of score improvement (how uncertain the benefit is)
    - Cost: rate increase from modifying this block

    The efficient frontier gives us optimal allocation of gradient magnitude
    across blocks. Blocks with high return/risk ratio get more gradient budget.

    Config:
        base_lr (float): base learning rate. Default 1.0.
        block_size (int): spatial block size for portfolio assets. Default 8.
        risk_free_rate (float): minimum acceptable improvement ratio. Default 0.0.
        ema_decay (float): EMA for return/risk estimation. Default 0.9.
        min_weight (float): minimum allocation per block (prevents starvation). Default 0.1.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.block_size = config.get("block_size", 8)
        self.risk_free_rate = config.get("risk_free_rate", 0.0)
        self.ema_decay = config.get("ema_decay", 0.9)
        self.min_weight = config.get("min_weight", 0.1)

        # Per-block running stats
        self._block_return_ema: torch.Tensor | None = None
        self._block_var_ema: torch.Tensor | None = None

    def _compute_block_stats(self, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-block expected return and risk from gradient.

        Return = mean |grad| in block (expected improvement magnitude).
        Risk = variance of |grad| in block (uncertainty of improvement).
        """
        # grad shape: (..., H, W, 3) or (..., H, W)
        if grad.dim() >= 3 and grad.shape[-1] == 3:
            # (N, H, W, 3) -> take norm over color channel
            g = grad.norm(dim=-1)  # (N, H, W) or (H, W)
        else:
            g = grad.abs()

        # Ensure at least 2D spatial
        while g.dim() < 2:
            g = g.unsqueeze(0)

        H, W = g.shape[-2], g.shape[-1]
        bs = self.block_size

        # Pad to multiple of block_size
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            g = F.pad(g, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # Reshape into blocks: (..., H//bs, bs, W//bs, bs)
        Hp, Wp = g.shape[-2], g.shape[-1]
        blocks = g.reshape(*g.shape[:-2], Hp // bs, bs, Wp // bs, bs)
        # Per-block return = mean gradient magnitude
        block_return = blocks.mean(dim=(-3, -1))  # (..., H//bs, W//bs)
        # Per-block risk = variance
        block_var = blocks.var(dim=(-3, -1)) + 1e-8

        return block_return, block_var

    def _sharpe_weights(
        self, block_return: torch.Tensor, block_var: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sharpe-ratio based weights (simplified efficient frontier)."""
        sharpe = (block_return - self.risk_free_rate) / (block_var.sqrt() + 1e-8)
        # Softmax over blocks to get allocation weights
        sharpe_flat = sharpe.reshape(-1)
        weights_flat = F.softmax(sharpe_flat, dim=0)
        weights = weights_flat.reshape(sharpe.shape)
        # Normalize so mean weight = 1, apply floor
        weights = weights / (weights.mean() + 1e-8)
        weights = weights.clamp(min=self.min_weight)
        return weights

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        block_return, block_var = self._compute_block_stats(grad)

        # EMA update
        d = self.ema_decay
        if self._block_return_ema is None:
            self._block_return_ema = block_return.detach()
            self._block_var_ema = block_var.detach()
        else:
            # Handle shape changes gracefully
            if self._block_return_ema.shape == block_return.shape:
                self._block_return_ema = d * self._block_return_ema + (1 - d) * block_return.detach()
                self._block_var_ema = d * self._block_var_ema + (1 - d) * block_var.detach()
            else:
                self._block_return_ema = block_return.detach()
                self._block_var_ema = block_var.detach()

        weights = self._sharpe_weights(self._block_return_ema, self._block_var_ema)

        # Upsample weights to pixel resolution
        bs = self.block_size
        # weights shape: (..., Hb, Wb) -> need (..., H, W, 1) or broadcast
        w_up = weights.unsqueeze(-1).unsqueeze(-1)  # (..., Hb, Wb, 1, 1)
        w_up = w_up.expand(*weights.shape, bs, bs)  # (..., Hb, Wb, bs, bs)
        # Merge block dims back to spatial
        Hb, Wb = weights.shape[-2], weights.shape[-1]
        w_spatial = w_up.reshape(*weights.shape[:-2], Hb * bs, Wb * bs)

        # Trim to original pixel size
        H, W = grad.shape[-3] if grad.dim() >= 3 and grad.shape[-1] == 3 else grad.shape[-2], \
               grad.shape[-2] if grad.dim() >= 3 and grad.shape[-1] == 3 else grad.shape[-1]
        if grad.dim() >= 3 and grad.shape[-1] == 3:
            H, W = grad.shape[-3], grad.shape[-2]
            w_spatial = w_spatial[..., :H, :W].unsqueeze(-1)  # (..., H, W, 1)
        else:
            H, W = grad.shape[-2], grad.shape[-1]
            w_spatial = w_spatial[..., :H, :W]

        lr_map = self.base_lr * w_spatial
        pixels = pixels - lr_map * grad
        return self._clamp_pixels(pixels)


# ── 5. Pairs Trading / Statistical Arbitrage ──────────────────────────


class PairsTradingOptimizer(FinanceOptimizer):
    """Mean-reversion between correlated pixel block pairs.

    Pairs trading: find cointegrated asset pairs, trade mean reversion.
    Here: find pixel block pairs with correlated scorer responses.
    When one diverges from its cointegrated partner, correct it toward
    the partner's level (it's "cheaper" to fix the divergence).

    We pair each block with its spatial neighbor and track the spread
    (difference in scorer gradient magnitude). When spread exceeds
    a threshold, apply corrective gradient toward the mean.

    Config:
        base_lr (float): base learning rate. Default 1.0.
        block_size (int): block size for pairing. Default 16.
        spread_threshold (float): z-score threshold to trigger correction. Default 1.5.
        correction_strength (float): how aggressively to correct divergence. Default 0.3.
        ema_decay (float): EMA for spread statistics. Default 0.95.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.block_size = config.get("block_size", 16)
        self.spread_threshold = config.get("spread_threshold", 1.5)
        self.correction_strength = config.get("correction_strength", 0.3)
        self.ema_decay = config.get("ema_decay", 0.95)

        self._spread_mean: torch.Tensor | None = None
        self._spread_var: torch.Tensor | None = None

    def _compute_block_means(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce to per-block means. Input: (..., H, W, C) -> (..., Hb, Wb, C)."""
        bs = self.block_size
        shape = tensor.shape
        H, W = shape[-3], shape[-2]
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor.permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor,
                          (0, pad_w, 0, pad_h) if tensor.dim() < 4 else (0, 0, 0, pad_w, 0, pad_h),
                          mode="constant", value=0.0)
            if tensor.dim() == 4 and shape[-1] == 3:
                tensor = tensor.permute(0, 2, 3, 1)

        Hp = H + pad_h
        Wp = W + pad_w
        if tensor.dim() >= 3 and tensor.shape[-1] in (1, 3):
            C = tensor.shape[-1]
            t = tensor.reshape(*shape[:-3], Hp // bs, bs, Wp // bs, bs, C)
            return t.mean(dim=(-4, -2))  # (..., Hb, Wb, C)
        else:
            t = tensor.reshape(*shape[:-2], Hp // bs, bs, Wp // bs, bs)
            return t.mean(dim=(-3, -1))

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration

        # Standard gradient step first
        pixels = pixels - self.base_lr * grad

        # Compute per-block pixel means for spread calculation
        if pixels.dim() >= 3 and pixels.shape[-1] == 3:
            block_means = self._compute_block_means(pixels.detach())

            # Horizontal pairs: spread = block[i,j] - block[i,j+1]
            if block_means.shape[-2] > 1:
                h_spread = block_means[..., :-1, :, :] - block_means[..., 1:, :, :]
                h_spread_norm = h_spread.norm(dim=-1)  # (..., Hb, Wb-1)

                # Track running spread statistics
                d = self.ema_decay
                if self._spread_mean is None or self._spread_mean.shape != h_spread_norm.shape:
                    self._spread_mean = h_spread_norm.detach().clone()
                    self._spread_var = torch.ones_like(h_spread_norm) * 0.01
                else:
                    self._spread_mean = d * self._spread_mean + (1 - d) * h_spread_norm.detach()
                    diff = h_spread_norm.detach() - self._spread_mean
                    self._spread_var = d * self._spread_var + (1 - d) * diff**2

                # Z-score of current spread
                z = (h_spread_norm - self._spread_mean) / (self._spread_var.sqrt() + 1e-8)

                # Where |z| > threshold, apply mean-reversion correction
                correction_mask = (z.abs() > self.spread_threshold).float()

                # Build per-pixel correction: pull divergent blocks toward their pair mean
                bs = self.block_size
                H, W = pixels.shape[-3], pixels.shape[-2]
                correction = torch.zeros_like(pixels)

                for bi in range(min(correction_mask.shape[-2], H // bs)):
                    for bj in range(min(correction_mask.shape[-1], W // bs - 1)):
                        if correction_mask.dim() == 2:
                            should_correct = correction_mask[bi, bj].item()
                        else:
                            should_correct = correction_mask[..., bi, bj].mean().item()
                        if should_correct > 0.5:
                            # Pull both blocks toward their mean
                            h_start = bi * bs
                            h_end = min(h_start + bs, H)
                            w1_start = bj * bs
                            w1_end = min(w1_start + bs, W)
                            w2_start = (bj + 1) * bs
                            w2_end = min(w2_start + bs, W)

                            mean_val = (
                                pixels[..., h_start:h_end, w1_start:w1_end, :].mean(dim=(-3, -2), keepdim=True)
                                + pixels[..., h_start:h_end, w2_start:w2_end, :].mean(dim=(-3, -2), keepdim=True)
                            ) / 2.0

                            correction[..., h_start:h_end, w1_start:w1_end, :] += (
                                self.correction_strength
                                * (mean_val - pixels[..., h_start:h_end, w1_start:w1_end, :])
                            )
                            correction[..., h_start:h_end, w2_start:w2_end, :] += (
                                self.correction_strength
                                * (mean_val - pixels[..., h_start:h_end, w2_start:w2_end, :])
                            )

                pixels = pixels + correction

        return self._clamp_pixels(pixels)


# ── 6. GARCH Volatility for Adaptive Noise ────────────────────────────


class GARCHVolatilityOptimizer(FinanceOptimizer):
    """GARCH(1,1) gradient volatility for Langevin dynamics.

    GARCH (Bollerslev 1986): sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

    Unlike Adam's simple EMA of squared gradients, GARCH properly models
    volatility clustering: large gradient changes predict more large changes.
    This is the correct statistical model for the heteroskedastic gradient
    noise we observe near sharp loss landscape features.

    We use sigma_t to scale Langevin exploration noise:
    - High predicted volatility -> larger exploration noise (landscape is rugged)
    - Low predicted volatility -> smaller noise (landscape is smooth, exploit)

    Config:
        base_lr (float): base learning rate. Default 1.0.
        omega (float): GARCH intercept (long-run variance). Default 0.01.
        alpha (float): GARCH ARCH coefficient (shock sensitivity). Default 0.15.
        beta (float): GARCH GARCH coefficient (persistence). Default 0.80.
        noise_scale (float): Langevin noise multiplier. Default 0.1.
        use_langevin (bool): whether to add exploration noise. Default True.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.omega = config.get("omega", 0.01)
        self.alpha_garch = config.get("alpha", 0.15)
        self.beta_garch = config.get("beta", 0.80)
        self.noise_scale = config.get("noise_scale", 0.1)
        self.use_langevin = config.get("use_langevin", True)

        # Validate stationarity: alpha + beta < 1
        assert self.alpha_garch + self.beta_garch < 1.0, (
            f"GARCH stationarity violated: alpha + beta = "
            f"{self.alpha_garch + self.beta_garch:.3f} >= 1.0"
        )

        self._sigma_sq: torch.Tensor | None = None
        self._prev_innovation_sq: torch.Tensor | None = None

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration

        # Innovation = gradient minus its EMA (the "shock")
        # For simplicity, use grad^2 as the squared innovation
        eps_sq = (grad**2).detach()

        if self._sigma_sq is None:
            # Initialize conditional variance at unconditional level
            unconditional = self.omega / (1.0 - self.alpha_garch - self.beta_garch + 1e-8)
            self._sigma_sq = torch.full_like(grad, unconditional).detach()
            self._prev_innovation_sq = eps_sq.clone()
        else:
            if self._sigma_sq.shape == grad.shape:
                # GARCH(1,1) update
                self._sigma_sq = (
                    self.omega
                    + self.alpha_garch * self._prev_innovation_sq
                    + self.beta_garch * self._sigma_sq
                ).detach()
            else:
                unconditional = self.omega / (1.0 - self.alpha_garch - self.beta_garch + 1e-8)
                self._sigma_sq = torch.full_like(grad, unconditional).detach()

        self._prev_innovation_sq = eps_sq.clone()

        # Adaptive LR: scale inversely with predicted volatility (like Adam)
        sigma = self._sigma_sq.sqrt() + 1e-8
        adaptive_lr = self.base_lr / sigma
        # Cap adaptive LR to prevent explosion in low-variance regions
        adaptive_lr = adaptive_lr.clamp(max=self.base_lr * 100.0)

        pixels = pixels - adaptive_lr * grad

        # Langevin noise scaled by predicted volatility
        if self.use_langevin and self.noise_scale > 0:
            noise = torch.randn_like(pixels) * self.noise_scale * sigma.sqrt()
            pixels = pixels + noise

        return self._clamp_pixels(pixels)

    def state_dict(self) -> dict[str, Any]:
        d = super().state_dict()
        d["garch_params"] = {
            "omega": self.omega,
            "alpha": self.alpha_garch,
            "beta": self.beta_garch,
        }
        return d


# ── 7. Order Book Dynamics / Level 2 Optimization ─────────────────────


class OrderBookOptimizer(FinanceOptimizer):
    """Priority queue pixel optimization based on order book dynamics.

    Model each pixel's potential change as a "limit order":
    - Price = rate cost of changing this pixel (approximated by change magnitude)
    - Size = score improvement (gradient magnitude)
    - Priority = size / price = gradient magnitude / rate cost

    Execute pixel changes in priority order (VWAP-like):
    - "Market orders": large changes to high-priority pixels (immediate benefit)
    - "Limit orders": small refinements to low-priority pixels (gradual)

    We implement this via top-k selection: only the top fraction of pixels
    (by priority score) get the full learning rate. Others get a reduced rate.

    Config:
        base_lr (float): learning rate for top-priority pixels. Default 1.0.
        background_lr_ratio (float): LR ratio for non-priority pixels. Default 0.1.
        top_fraction (float): fraction of pixels treated as market orders. Default 0.3.
        rate_cost_weight (float): how much to penalize large changes. Default 0.01.
        requeue_every (int): re-sort priorities every N steps. Default 5.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.background_lr_ratio = config.get("background_lr_ratio", 0.1)
        self.top_fraction = config.get("top_fraction", 0.3)
        self.rate_cost_weight = config.get("rate_cost_weight", 0.01)
        self.requeue_every = config.get("requeue_every", 5)

        self._priority_mask: torch.Tensor | None = None
        self._prev_pixels: torch.Tensor | None = None

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration

        recompute = (
            iteration % self.requeue_every == 0
            or self._priority_mask is None
            or self._priority_mask.shape != grad.shape
        )

        if recompute:
            # Score improvement = gradient magnitude
            score_benefit = grad.abs()

            # Rate cost approximation: how much this pixel differs from neighbors
            # (changes to "isolated" pixels cost more in rate)
            if self._prev_pixels is not None and self._prev_pixels.shape == pixels.shape:
                change_magnitude = (pixels - self._prev_pixels).abs().detach()
            else:
                change_magnitude = torch.ones_like(grad) * 0.01

            rate_cost = change_magnitude * self.rate_cost_weight + 1e-8

            # Priority = benefit / cost
            priority = score_benefit / rate_cost

            # Top-k mask
            flat_priority = priority.reshape(-1)
            k = max(1, int(self.top_fraction * flat_priority.numel()))
            threshold = torch.topk(flat_priority, k, largest=True).values[-1]
            self._priority_mask = (priority >= threshold).float()

        self._prev_pixels = pixels.detach().clone()

        # Apply differentiated LR
        lr_map = (
            self._priority_mask * self.base_lr
            + (1 - self._priority_mask) * self.base_lr * self.background_lr_ratio
        )
        pixels = pixels - lr_map * grad
        return self._clamp_pixels(pixels)


# ── 8. Avellaneda-Stoikov Market Making ────────────────────────────────


class AvellanedaStoikovOptimizer(FinanceOptimizer):
    """Market-making inspired perturbation range control.

    Avellaneda & Stoikov (2008): set optimal bid/ask spread around fair value
    based on inventory risk and volatility.

    Our adaptation:
    - Fair value = current best pixel values (local optimum estimate)
    - Inventory = distance from constraint satisfaction
    - Spread = allowed perturbation range, narrows as we approach optimum

    The reservation price r = s - q * gamma * sigma^2 * (T-t)
    where q = inventory, gamma = risk aversion, sigma = volatility.

    We use this to compute per-pixel perturbation bounds that shrink
    over time and near constraint boundaries.

    Config:
        base_lr (float): base learning rate. Default 1.0.
        gamma (float): risk aversion (controls spread tightening). Default 0.1.
        initial_spread (float): initial perturbation range. Default 10.0.
        min_spread (float): minimum perturbation range. Default 0.5.
        total_steps (int): total optimization steps. Default 1000.
        inventory_decay (float): EMA for inventory tracking. Default 0.95.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.gamma = config.get("gamma", 0.1)
        self.initial_spread = config.get("initial_spread", 10.0)
        self.min_spread = config.get("min_spread", 0.5)
        self.total_steps = config.get("total_steps", 1000)
        self.inventory_decay = config.get("inventory_decay", 0.95)

        self._fair_value: torch.Tensor | None = None
        self._inventory: torch.Tensor | None = None
        self._volatility: torch.Tensor | None = None
        self._prev_grad: torch.Tensor | None = None

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        t_frac = iteration / max(self.total_steps, 1)
        time_remaining = max(1.0 - t_frac, 0.01)

        # Update fair value (EMA of pixel values)
        d = self.inventory_decay
        if self._fair_value is None:
            self._fair_value = pixels.detach().clone()
            self._inventory = torch.zeros_like(pixels)
            self._volatility = grad.abs().detach().clone()
        else:
            self._fair_value = d * self._fair_value + (1 - d) * pixels.detach()
            # Inventory = cumulative signed gradient (how far we've moved from fair value)
            self._inventory = d * self._inventory + (1 - d) * grad.detach().sign()

        # Volatility estimate from gradient changes
        if self._prev_grad is not None and self._prev_grad.shape == grad.shape:
            grad_change = (grad - self._prev_grad).abs().detach()
            self._volatility = d * self._volatility + (1 - d) * grad_change
        self._prev_grad = grad.detach().clone()

        # Avellaneda-Stoikov spread: spread = gamma * sigma^2 * (T-t) + 2/gamma * ln(1 + gamma/k)
        # Simplified: spread = initial_spread * (1 - inventory_pressure) * time_remaining
        sigma_sq = self._volatility**2 + 1e-8
        inventory_pressure = self._inventory.abs().clamp(max=0.9)
        spread = (
            self.initial_spread * (1.0 - inventory_pressure) * time_remaining
            + self.gamma * sigma_sq * time_remaining
        ).clamp(min=self.min_spread)

        # Gradient step with spread-limited perturbation
        step = self.base_lr * grad
        step = step.clamp(-spread, spread)
        pixels = pixels - step

        return self._clamp_pixels(pixels)


# ── 9. Momentum / Mean Reversion Hybrid ───────────────────────────────


class MomentumMeanReversionOptimizer(FinanceOptimizer):
    """Hybrid momentum + mean reversion with regime detection.

    In HFT: momentum strategies work in trending markets, mean-reversion
    works in range-bound markets. The key is detecting the regime switch.

    Here:
    - Momentum phase: follow gradient direction (like Adam/SGD momentum)
    - Mean reversion phase: pull toward running average (like Polyak/SWA)
    - Regime detection: track gradient autocorrelation
        - Positive autocorrelation -> trending -> use momentum
        - Negative autocorrelation -> mean-reverting -> use averaging

    The crossover signal is the sign of the gradient autocorrelation:
        rho_t = EMA(grad_t * grad_{t-1}) / (EMA(grad_t^2) + eps)

    Config:
        base_lr (float): base learning rate. Default 1.0.
        momentum_coeff (float): momentum coefficient (like beta1 in Adam). Default 0.9.
        reversion_strength (float): strength of pull toward running mean. Default 0.1.
        autocorr_ema (float): EMA decay for autocorrelation tracking. Default 0.95.
        crossover_threshold (float): |rho| below this -> use hybrid. Default 0.2.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.momentum_coeff = config.get("momentum_coeff", 0.9)
        self.reversion_strength = config.get("reversion_strength", 0.1)
        self.autocorr_ema = config.get("autocorr_ema", 0.95)
        self.crossover_threshold = config.get("crossover_threshold", 0.2)

        self._velocity: torch.Tensor | None = None
        self._running_mean: torch.Tensor | None = None
        self._prev_grad: torch.Tensor | None = None
        self._cross_product_ema: torch.Tensor | None = None
        self._grad_sq_ema: torch.Tensor | None = None

    def _compute_regime(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute per-element regime indicator: +1 = momentum, -1 = mean reversion."""
        d = self.autocorr_ema

        if self._prev_grad is None or self._prev_grad.shape != grad.shape:
            self._prev_grad = grad.detach().clone()
            self._cross_product_ema = torch.zeros_like(grad)
            self._grad_sq_ema = (grad**2).detach() + 1e-8
            return torch.ones_like(grad)  # Default to momentum

        cross = (grad * self._prev_grad).detach()
        grad_sq = (grad**2).detach()

        self._cross_product_ema = d * self._cross_product_ema + (1 - d) * cross
        self._grad_sq_ema = d * self._grad_sq_ema + (1 - d) * grad_sq

        # Autocorrelation
        rho = self._cross_product_ema / (self._grad_sq_ema + 1e-8)
        self._prev_grad = grad.detach().clone()

        return rho

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration

        rho = self._compute_regime(grad)

        # Momentum component
        if self._velocity is None or self._velocity.shape != grad.shape:
            self._velocity = grad.detach().clone()
        else:
            self._velocity = self.momentum_coeff * self._velocity + (1 - self.momentum_coeff) * grad.detach()

        # Running mean for mean reversion target
        if self._running_mean is None or self._running_mean.shape != pixels.shape:
            self._running_mean = pixels.detach().clone()
        else:
            self._running_mean = 0.99 * self._running_mean + 0.01 * pixels.detach()

        # Regime-dependent step
        # momentum_weight in [0, 1]: high when rho > threshold (trending)
        momentum_weight = ((rho - self.crossover_threshold) / 0.5).sigmoid()

        # Momentum step: follow velocity
        momentum_step = self.base_lr * self._velocity

        # Mean reversion step: pull toward running mean
        reversion_step = self.reversion_strength * (self._running_mean - pixels)

        # Blend based on regime
        total_step = momentum_weight * momentum_step + (1 - momentum_weight) * reversion_step.detach()
        pixels = pixels - total_step

        return self._clamp_pixels(pixels)


# ── 10. Risk Parity ──────────────────────────────────────────────────


class RiskParityOptimizer(FinanceOptimizer):
    """Equal risk contribution across score components.

    Risk Parity (Qian 2005): allocate so each asset contributes equal
    portfolio risk (measured by marginal risk contribution).

    Our score: S = 100*seg + sqrt(10*pose) + 25*rate
    Three "assets" with wildly different scales and sensitivities.

    Without risk parity, one component (usually SegNet with its 100x weight)
    dominates the gradient. Risk parity ensures each component contributes
    equally to marginal score improvement.

    Implementation:
    - Track running variance of each component's gradient contribution
    - Scale each component's gradient inversely with its risk contribution
    - This is equivalent to the analytical risk parity portfolio:
      w_i proportional to 1/sigma_i when correlations are moderate

    Config:
        base_lr (float): base learning rate. Default 1.0.
        ema_decay (float): EMA for component variance tracking. Default 0.95.
        seg_weight (float): initial seg gradient weight. Default 1.0.
        pose_weight (float): initial pose gradient weight. Default 1.0.
        rate_weight (float): initial rate gradient weight. Default 1.0.
        rebalance_every (int): recompute weights every N steps. Default 10.

    Note: this optimizer expects split gradients. If you only have the total
    gradient, it falls back to a simpler per-element risk parity.
    """

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.base_lr = config.get("base_lr", 1.0)
        self.ema_decay = config.get("ema_decay", 0.95)
        self.seg_weight = config.get("seg_weight", 1.0)
        self.pose_weight = config.get("pose_weight", 1.0)
        self.rate_weight = config.get("rate_weight", 1.0)
        self.rebalance_every = config.get("rebalance_every", 10)

        # Running variance of gradient per spatial element
        self._grad_var_ema: torch.Tensor | None = None
        self._grad_mean_ema: torch.Tensor | None = None

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration
        d = self.ema_decay

        # Per-element risk parity (works even without split gradients)
        grad_sq = (grad**2).detach()
        if self._grad_var_ema is None or self._grad_var_ema.shape != grad.shape:
            self._grad_var_ema = grad_sq.clone()
            self._grad_mean_ema = grad.detach().clone()
        else:
            self._grad_var_ema = d * self._grad_var_ema + (1 - d) * grad_sq
            self._grad_mean_ema = d * self._grad_mean_ema + (1 - d) * grad.detach()

        # Marginal risk = sigma_i = sqrt(E[g^2] - E[g]^2)
        variance = (self._grad_var_ema - self._grad_mean_ema**2).clamp(min=1e-12)
        sigma = variance.sqrt()

        # Risk parity weight: w_i proportional to 1/sigma_i
        inv_sigma = 1.0 / (sigma + 1e-8)
        # Normalize so mean weight = 1
        rp_weight = inv_sigma / (inv_sigma.mean() + 1e-8)
        # Clamp to prevent extreme values
        rp_weight = rp_weight.clamp(0.01, 100.0)

        lr_map = self.base_lr * rp_weight
        pixels = pixels - lr_map * grad
        return self._clamp_pixels(pixels)

    def step_with_components(
        self,
        pixels: torch.Tensor,
        seg_grad: torch.Tensor,
        pose_grad: torch.Tensor,
        rate_grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Step with split gradients — proper 3-component risk parity.

        This is the preferred interface when you have separate gradients
        for SegNet, PoseNet, and rate components.
        """
        self.iteration = iteration
        d = self.ema_decay

        # Track per-component variance
        seg_var = (seg_grad**2).detach().mean()
        pose_var = (pose_grad**2).detach().mean()
        rate_var = (rate_grad**2).detach().mean()

        # Risk parity: weight inversely proportional to volatility
        inv_seg = 1.0 / (seg_var.sqrt() + 1e-8)
        inv_pose = 1.0 / (pose_var.sqrt() + 1e-8)
        inv_rate = 1.0 / (rate_var.sqrt() + 1e-8)

        # Normalize weights to sum to 3 (one unit per component)
        total_inv = inv_seg + inv_pose + inv_rate + 1e-8
        w_seg = 3.0 * inv_seg / total_inv
        w_pose = 3.0 * inv_pose / total_inv
        w_rate = 3.0 * inv_rate / total_inv

        combined_grad = w_seg * seg_grad + w_pose * pose_grad + w_rate * rate_grad
        pixels = pixels - self.base_lr * combined_grad
        return self._clamp_pixels(pixels)


# ── Ensemble / Pipeline ───────────────────────────────────────────────


class FinanceEnsembleOptimizer(FinanceOptimizer):
    """Ensemble of finance optimizers with configurable mixing.

    Runs multiple finance optimizers and blends their proposed updates.
    Like a multi-strategy hedge fund: diversification across alpha sources.

    Config:
        optimizers (list[dict]): list of {"name": str, "weight": float, "config": dict}
        blend_mode (str): "weighted_average" | "best_sharpe" | "round_robin". Default "weighted_average".
        base_lr (float): global LR override. Default 1.0.
    """

    REGISTRY: dict[str, type[CrossDisciplinaryOptimizer]] = {
        "almgren_chriss": AlmgrenChrissOptimizer,
        "kelly": KellyCriterionOptimizer,
        "implied_vol": ImpliedVolatilityOptimizer,
        "markowitz": MarkowitzOptimizer,
        "pairs_trading": PairsTradingOptimizer,
        "garch": GARCHVolatilityOptimizer,
        "order_book": OrderBookOptimizer,
        "avellaneda_stoikov": AvellanedaStoikovOptimizer,
        "momentum_reversion": MomentumMeanReversionOptimizer,
        "risk_parity": RiskParityOptimizer,
    }

    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__(config, device)
        self.blend_mode = config.get("blend_mode", "weighted_average")

        self.sub_optimizers: list[tuple[float, CrossDisciplinaryOptimizer]] = []
        for spec in config.get("optimizers", []):
            name = spec["name"]
            weight = spec.get("weight", 1.0)
            sub_config = spec.get("config", {})
            cls = self.REGISTRY[name]
            self.sub_optimizers.append((weight, cls(sub_config, device)))

        if not self.sub_optimizers:
            raise ValueError("FinanceEnsemble requires at least one sub-optimizer")

    def step(
        self,
        pixels: torch.Tensor,
        grad: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        self.iteration = iteration

        if self.blend_mode == "round_robin":
            idx = iteration % len(self.sub_optimizers)
            _, opt = self.sub_optimizers[idx]
            return opt.step(pixels, grad, iteration)

        # Compute proposed updates from each optimizer
        proposals = []
        for weight, opt in self.sub_optimizers:
            proposed = opt.step(pixels.clone(), grad, iteration)
            proposals.append((weight, proposed))

        if self.blend_mode == "weighted_average":
            total_weight = sum(w for w, _ in proposals)
            blended = sum(w * p for w, p in proposals) / (total_weight + 1e-8)
            return self._clamp_pixels(blended)

        # Fallback: equal weight average
        blended = sum(p for _, p in proposals) / len(proposals)
        return self._clamp_pixels(blended)


# ── Smoke tests ───────────────────────────────────────────────────────


def smoke_test_all(device: str = "cpu") -> dict[str, bool]:
    """Run smoke tests for all finance optimizers.

    Creates a small (32, 32, 3) pixel tensor and runs 5 steps of each
    optimizer, verifying that:
    1. Output shape matches input shape
    2. Output is in [0, 255]
    3. No NaN/Inf values
    4. Pixels actually changed (optimizer did something)

    Returns dict mapping optimizer name to pass/fail.
    """
    results: dict[str, bool] = {}
    torch.manual_seed(42)
    H, W, C = 32, 32, 3
    pixels = torch.rand(H, W, C, device=device) * 255.0
    pixels.requires_grad_(False)

    test_configs: dict[str, tuple[type, dict]] = {
        "almgren_chriss": (AlmgrenChrissOptimizer, {"total_steps": 20, "base_lr": 0.5}),
        "kelly": (KellyCriterionOptimizer, {"base_lr": 0.5}),
        "implied_vol": (ImpliedVolatilityOptimizer, {"base_lr": 0.5}),
        "markowitz": (MarkowitzOptimizer, {"base_lr": 0.5, "block_size": 8}),
        "pairs_trading": (PairsTradingOptimizer, {"base_lr": 0.5, "block_size": 8}),
        "garch": (GARCHVolatilityOptimizer, {"base_lr": 0.5, "alpha": 0.1, "beta": 0.8}),
        "order_book": (OrderBookOptimizer, {"base_lr": 0.5}),
        "avellaneda_stoikov": (AvellanedaStoikovOptimizer, {"base_lr": 0.5, "total_steps": 20}),
        "momentum_reversion": (MomentumMeanReversionOptimizer, {"base_lr": 0.5}),
        "risk_parity": (RiskParityOptimizer, {"base_lr": 0.5}),
        "ensemble": (
            FinanceEnsembleOptimizer,
            {
                "optimizers": [
                    {"name": "almgren_chriss", "weight": 1.0, "config": {"total_steps": 20, "base_lr": 0.5}},
                    {"name": "kelly", "weight": 1.0, "config": {"base_lr": 0.5}},
                ],
                "blend_mode": "weighted_average",
            },
        ),
    }

    for name, (cls, cfg) in test_configs.items():
        try:
            opt = cls(cfg, device=device)
            p = pixels.clone()
            initial = p.clone()
            for i in range(5):
                grad = torch.randn_like(p) * 10.0
                p = opt.step(p, grad, i)

            ok = (
                p.shape == pixels.shape
                and p.min() >= 0.0
                and p.max() <= 255.0
                and not torch.isnan(p).any()
                and not torch.isinf(p).any()
                and (p - initial).abs().sum() > 0.01
            )
            results[name] = bool(ok)
        except Exception as e:
            results[name] = False
            print(f"  FAIL {name}: {e}")

    return results


# ── Yousfi + Contrarian Curated Picks ─────────────────────────────────


def yousfi_contrarian_picks() -> dict[str, Any]:
    """Yousfi and Contrarian's curated picks for experimentation.

    HONEST assessment of which cross-disciplinary algorithms are worth trying
    for our specific problem: optimizing 384x512x3 pixels to minimize
    S = 100*seg + sqrt(10*pose) + 25*rate against frozen scorer networks.

    Returns dict with keys:
        'must_try': algorithms that could change our score significantly
        'worth_trying': interesting, worth a smoke test
        'skip': theoretically sound but unlikely to help our specific problem
        'reasoning': dict mapping algorithm name to why it's in that tier
    """
    return {
        "must_try": [
            # --- Finance ---
            "Risk Parity (finance)",
            "Order Book / VWAP Priority Queue (finance)",
            "Implied Volatility Heat Map (finance)",
            # --- Physics ---
            "Simulated Annealing (physics)",
        ],
        "worth_trying": [
            # --- Finance ---
            "Almgren-Chriss Execution Schedule (finance)",
            "Kelly Criterion Step Sizing (finance)",
            "GARCH Volatility (finance)",
            "Momentum/Mean-Reversion Hybrid (finance)",
            # --- Physics ---
            "Hamiltonian Monte Carlo (physics)",
            # --- Biology ---
            "CMA-ES / Evolutionary Strategy (biology)",
            # --- Chemistry ---
            "Basin Hopping (chemistry/molecular)",
        ],
        "skip": [
            # --- Finance ---
            "Markowitz Portfolio (finance)",
            "Pairs Trading (finance)",
            "Avellaneda-Stoikov Market Making (finance)",
            # --- Physics ---
            "Quantum Annealing (physics)",
            "Replica Exchange / Parallel Tempering (physics)",
            "Langevin Dynamics plain (physics)",
            # --- Biology ---
            "Genetic Algorithms (biology)",
            "Ant Colony Optimization (biology)",
            # --- Geophysics ---
            "Full Waveform Inversion (geophysics)",
            # --- Climate ---
            "Ensemble Kalman Filter (climate)",
            "Data Assimilation (climate)",
            # --- Astrophysics ---
            "Nested Sampling (astrophysics)",
            "Bayesian Optimization (astrophysics/ML)",
            # --- Chemistry ---
            "Nudged Elastic Band (chemistry)",
        ],
        "reasoning": {
            # ===== MUST TRY =====
            "Risk Parity (finance)": (
                "YOUSFI: This is the single most obviously useful transfer. Our score "
                "S = 100*seg + sqrt(10*pose) + 25*rate has three components with wildly "
                "different gradient scales. Without risk parity, SegNet's 100x coefficient "
                "drowns PoseNet's sqrt(10*x) gradient. We literally measured this problem: "
                "PoseNet sensitivity is the bottleneck (117% of gains come from PoseNet). "
                "Risk parity is the correct fix.\n"
                "CONTRARIAN: Agree. This is not even speculative. It's the analytical "
                "solution to the multi-objective scaling problem we already know we have. "
                "The only question is whether the improvement over manual weight tuning "
                "(segnet_loss_weight=30) is significant. I bet yes."
            ),
            "Order Book / VWAP Priority Queue (finance)": (
                "YOUSFI: Every pixel is not equally important. We know from saliency maps "
                "that road boundaries and moving objects dominate the score. The order book "
                "metaphor gives us an elegant priority queue: process high-scorer-impact "
                "pixels first. This is rate-aware gradient descent — exactly what we need "
                "since rate is 25x in the score formula.\n"
                "CONTRARIAN: The insight is solid: don't waste gradient budget on sky pixels "
                "the scorer ignores. But we already have boundary_weight and hard_frame_ratio "
                "doing something similar. The question is whether order-book framing adds "
                "anything over those. Worth testing to find out."
            ),
            "Implied Volatility Heat Map (finance)": (
                "YOUSFI: The per-pixel volatility map is EXACTLY the sensitivity analysis "
                "we need. High IV = scorer cares about this pixel. Low IV = scorer is "
                "indifferent. This maps directly to our Trick 13 (PoseNet blind spots) "
                "and Trick 14 (SegNet skip connections). The IV map would discover these "
                "automatically instead of hand-coding them.\n"
                "CONTRARIAN: I like that this is self-discovering rather than hand-coded. "
                "The concern is computational cost — maintaining per-pixel running stats "
                "for 384*512*3 = 589,824 elements at every step. But the EMA implementation "
                "is O(1) per step, so it's fine."
            ),
            "Simulated Annealing (physics)": (
                "YOUSFI: Our loss landscape has many local minima (the scorer networks are "
                "deep CNNs). SA is the simplest way to escape them. The temperature schedule "
                "maps directly to the Almgren-Chriss urgency parameter — aggressive early, "
                "conservative late.\n"
                "CONTRARIAN: SA is well-understood, low-risk, and easy to combine with "
                "gradient descent (gradient + noise with temperature). The only risk is "
                "that pure SA without gradients is too slow for 589K-dimensional optimization. "
                "But gradient-guided SA (Langevin dynamics with annealing) is standard and "
                "should work. Must try."
            ),
            # ===== WORTH TRYING =====
            "Almgren-Chriss Execution Schedule (finance)": (
                "YOUSFI: The execution schedule insight is real — we do want to move fast "
                "early and slow late. But so does every cosine/linear LR schedule. The AC "
                "schedule is specifically optimal for minimizing market impact (overshooting), "
                "which maps to our problem of overshooting past sharp scorer minima.\n"
                "CONTRARIAN: The sinh-based schedule is a specific shape that MAY be better "
                "than cosine for our landscape. But 'may' is the key word. It's cheap to "
                "test (just a schedule), so worth a smoke test. Not a game-changer though."
            ),
            "Kelly Criterion Step Sizing (finance)": (
                "YOUSFI: Adaptive step sizing based on gradient SNR is genuinely useful. "
                "Near sharp minima, gradients are noisy and Kelly says slow down. On smooth "
                "plateaus, gradients are reliable and Kelly says go aggressive. This is "
                "what Adam tries to do but Kelly has the optimal sizing formula.\n"
                "CONTRARIAN: Adam works well in practice. Kelly's advantage is theoretical "
                "optimality, but the assumptions (independent bets, known probabilities) "
                "don't hold for correlated pixel gradients. Worth testing but I doubt "
                "it beats well-tuned Adam by much."
            ),
            "GARCH Volatility (finance)": (
                "YOUSFI: GARCH predicts heteroskedastic volatility clustering. In our "
                "problem, gradient noise IS heteroskedastic — it spikes at epoch boundaries "
                "and when hard frames rotate in. GARCH could pre-empt these spikes.\n"
                "CONTRARIAN: The alpha+beta stationarity constraint limits expressiveness. "
                "Adam's EMA of squared gradients is a degenerate GARCH(1,1) with alpha=0. "
                "The question is whether the ARCH term (shock sensitivity) adds anything "
                "over the GARCH term (persistence) that Adam already has. Marginal benefit."
            ),
            "Momentum/Mean-Reversion Hybrid (finance)": (
                "YOUSFI: The regime detection via gradient autocorrelation is clever. If "
                "we can detect when the optimizer is trending vs. oscillating, we can switch "
                "strategies. This might help with the PoseNet/SegNet tug-of-war.\n"
                "CONTRARIAN: SWA (which we already use) is basically the mean-reversion "
                "phase. And Adam momentum is the momentum phase. This hybrid's value "
                "depends on whether the autocorrelation crossover signal is reliable. "
                "Worth testing, but may just reinvent what Adam+SWA already does."
            ),
            "Hamiltonian Monte Carlo (physics)": (
                "YOUSFI: HMC uses gradient information to make efficient proposals in "
                "high-dimensional spaces. For 589K dimensions, random-walk MCMC is dead "
                "but HMC works because it follows the gradient flow.\n"
                "CONTRARIAN: HMC is expensive (leapfrog integration requires multiple "
                "gradient evaluations per step). Our gradient is already expensive (full "
                "scorer forward+backward). But even a poor-man's HMC (2-3 leapfrog steps) "
                "could help escape the sharp SegNet minima we're stuck in. Worth trying."
            ),
            "CMA-ES / Evolutionary Strategy (biology)": (
                "YOUSFI: CMA-ES is the gold standard for derivative-free optimization in "
                "moderate dimensions. Our trick: use CMA-ES on a low-dimensional "
                "parameterization (e.g., 64-dim latent of our postfilter) rather than raw "
                "pixels. That makes the covariance matrix tractable.\n"
                "CONTRARIAN: Agree on the latent-space approach. Raw 589K CMA-ES is "
                "impossible. But latent-space CMA-ES (64 dims) is exactly the right "
                "dimensionality for CMA-ES. Worth trying as a hyperparameter optimizer "
                "if nothing else."
            ),
            "Basin Hopping (chemistry/molecular)": (
                "YOUSFI: Basin hopping = local minimization + random perturbation + "
                "Metropolis acceptance. Simple, effective for funnel-shaped landscapes. "
                "Our landscape is probably funnel-shaped (many local minima near the "
                "global basin).\n"
                "CONTRARIAN: This is basically SA with local optimization steps. If SA "
                "is in must-try, basin hopping is the fancy version that might work "
                "better but costs more per iteration. Worth trying after SA."
            ),
            # ===== SKIP =====
            "Markowitz Portfolio (finance)": (
                "YOUSFI: Elegant theory, but the block-level portfolio optimization "
                "requires estimating a covariance matrix over thousands of pixel blocks. "
                "The covariance estimate will be terrible with our small effective sample "
                "size (iterations). Markowitz is famous for being estimation-error-sensitive.\n"
                "CONTRARIAN: Classic Markowitz requires inverting the covariance matrix, "
                "which with N=3000 blocks is O(N^3). Even the simplified Sharpe-ratio "
                "version is just a priority queue with extra steps. The order-book approach "
                "does the same thing more efficiently. Skip."
            ),
            "Pairs Trading (finance)": (
                "YOUSFI: The cointegration idea between adjacent pixel blocks is "
                "interesting in theory. But the per-block loop implementation is O(N^2) "
                "and the correction is just spatial smoothing — which a 3x3 Gaussian "
                "blur does faster and better.\n"
                "CONTRARIAN: Agreed. Pairs trading between pixel blocks is just TV "
                "regularization with extra steps. We already have total variation "
                "penalty in the constrained generator. Skip."
            ),
            "Avellaneda-Stoikov Market Making (finance)": (
                "YOUSFI: The bid-ask spread analogy is poetic but the practical effect "
                "is just gradient clipping with a time-decaying threshold. We already "
                "do gradient clipping.\n"
                "CONTRARIAN: Skip. This is the most over-fitted analogy in the set. "
                "Market making is about providing liquidity to other agents. We don't "
                "have other agents. The 'spread' is just an adaptive clipping range."
            ),
            "Quantum Annealing (physics)": (
                "CONTRARIAN: On classical hardware, 'quantum annealing' is just "
                "simulated annealing with a different noise kernel. No advantage over "
                "SA unless you have an actual quantum computer. Skip."
            ),
            "Replica Exchange / Parallel Tempering (physics)": (
                "YOUSFI: Elegant for multimodal distributions, but requires running "
                "multiple replicas at different temperatures simultaneously. Each replica "
                "needs a full scorer forward pass. We're already GPU-memory-constrained.\n"
                "CONTRARIAN: 4-8 replicas * scorer cost = impractical. Skip."
            ),
            "Langevin Dynamics plain (physics)": (
                "CONTRARIAN: Already subsumed by GARCH volatility optimizer which adds "
                "Langevin noise with proper volatility modeling. No reason to implement "
                "plain Langevin separately."
            ),
            "Genetic Algorithms (biology)": (
                "CONTRARIAN: Population-based methods in 589K dimensions are dead on "
                "arrival. You'd need a population of thousands, each requiring a scorer "
                "forward pass. CMA-ES on the latent space is the right evolutionary "
                "approach. Skip plain GA."
            ),
            "Ant Colony Optimization (biology)": (
                "CONTRARIAN: Discrete combinatorial optimizer. Our problem is continuous "
                "pixel optimization. Complete mismatch. Skip."
            ),
            "Full Waveform Inversion (geophysics)": (
                "YOUSFI: FWI is adjoint-state gradient descent with frequency continuation "
                "(start with low-frequency, refine). Interesting idea for our multi-scale "
                "problem but the 'wave equation' analogy doesn't map cleanly to CNNs.\n"
                "CONTRARIAN: FWI's practical magic is frequency continuation, which is "
                "just curriculum learning (coarse-to-fine). We already do this with "
                "hard_frame_ratio. No unique insight. Skip."
            ),
            "Ensemble Kalman Filter (climate)": (
                "YOUSFI: EnKF is great for sequential state estimation with nonlinear "
                "observations. But our problem is optimization, not filtering. We don't "
                "have a time-evolving state.\n"
                "CONTRARIAN: EnKF requires ensemble members (30-100 typically). Same "
                "GPU cost problem as parallel tempering. Skip."
            ),
            "Data Assimilation (climate)": (
                "CONTRARIAN: This is just 'combine model predictions with observations' "
                "which is... gradient descent. We're already doing it. Skip."
            ),
            "Nested Sampling (astrophysics)": (
                "YOUSFI: Nested sampling is for Bayesian evidence computation, not "
                "optimization. Its optimization version (ns-opt) exists but requires "
                "maintaining live points in 589K dimensions. Dead on arrival.\n"
                "CONTRARIAN: Skip. Wrong problem class."
            ),
            "Bayesian Optimization (astrophysics/ML)": (
                "YOUSFI: BO is excellent for expensive black-box optimization in <20 "
                "dimensions. Our problem is 589K dimensions. The Gaussian process "
                "surrogate can't scale.\n"
                "CONTRARIAN: Could work on the 5-10 dim hyperparameter space (LR, "
                "segnet_loss_weight, etc.) but NOT for pixel optimization. For "
                "hyperparameter tuning, just use grid search — we have so few knobs "
                "that BO's overhead isn't justified. Skip for pixels."
            ),
            "Nudged Elastic Band (chemistry)": (
                "CONTRARIAN: NEB finds minimum energy paths between two known states. "
                "We don't have two known states — we have one starting point and want "
                "the minimum. Wrong problem structure. Skip."
            ),
        },
        "meta_commentary": {
            "yousfi": (
                "The three must-try algorithms address the three biggest inefficiencies "
                "in our current optimizer: (1) risk parity fixes the multi-scale gradient "
                "problem we KNOW we have, (2) order book prioritizes pixels by scorer "
                "impact, (3) implied volatility discovers scorer sensitivity automatically. "
                "These aren't speculative — they're direct fixes for measured problems. "
                "Everything else is exploration, and most of it won't beat well-tuned Adam."
            ),
            "contrarian": (
                "I'll be blunt: most cross-disciplinary transfers are intellectual "
                "entertainment, not practical tools. The useful ones succeed because they "
                "address a STRUCTURAL mismatch between standard optimizers and our problem. "
                "Risk parity addresses the 100x/sqrt(10)/25x scale mismatch. Order book "
                "addresses the spatial non-uniformity. IV heat map addresses the "
                "heterogeneous sensitivity. Everything else is either (a) already done by "
                "Adam+SWA, (b) too expensive for 589K dimensions, or (c) solving the wrong "
                "problem. Don't let cool math distract from the leaderboard."
            ),
        },
    }


# ── Module-level convenience ──────────────────────────────────────────


OPTIMIZER_REGISTRY = FinanceEnsembleOptimizer.REGISTRY


def get_optimizer(name: str, config: dict[str, Any], device: str = "cpu") -> CrossDisciplinaryOptimizer:
    """Get a finance optimizer by name.

    Args:
        name: one of the keys in OPTIMIZER_REGISTRY.
        config: optimizer-specific configuration dict.
        device: torch device string.

    Returns:
        Initialized optimizer instance.
    """
    if name == "ensemble":
        return FinanceEnsembleOptimizer(config, device)
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    return OPTIMIZER_REGISTRY[name](config, device)


if __name__ == "__main__":
    print("Running finance optimizer smoke tests...")
    results = smoke_test_all()
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
    all_pass = all(results.values())
    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED.'}")

    print("\nYousfi + Contrarian picks:")
    picks = yousfi_contrarian_picks()
    print(f"  Must try ({len(picks['must_try'])}): {picks['must_try']}")
    print(f"  Worth trying ({len(picks['worth_trying'])}): {picks['worth_trying']}")
    print(f"  Skip ({len(picks['skip'])}): {picks['skip']}")
