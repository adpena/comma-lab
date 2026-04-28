from __future__ import annotations

import math

import torch
import torch.nn as nn


def _ks_uniform_pvalue(samples: torch.Tensor) -> float:
    """One-sample KS p-value approximation for U(0, 1)."""
    x = samples.detach().cpu().sort().values.double()
    n = x.numel()
    empirical_hi = torch.arange(1, n + 1, dtype=torch.double) / n
    empirical_lo = torch.arange(0, n, dtype=torch.double) / n
    d_plus = (empirical_hi - x).max().item()
    d_minus = (x - empirical_lo).max().item()
    d = max(d_plus, d_minus)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 0.0
    for k in range(1, 128):
        term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, p))


def test_sample_sigmas_distribution() -> None:
    from tac.self_augmentation_v2 import HighSigmaSampler, HighSigmaStrategyConfig

    cfg = HighSigmaStrategyConfig(redraw_fraction=0.05)
    gen = torch.Generator().manual_seed(1)
    sigmas = HighSigmaSampler(cfg, generator=gen).sample_sigmas(10_000, torch.device("cpu"))

    high = (sigmas >= cfg.high_sigma_min).float().mean().item()
    sigma = math.sqrt(cfg.redraw_fraction * (1.0 - cfg.redraw_fraction) / sigmas.numel())

    assert abs(high - cfg.redraw_fraction) <= 2.0 * sigma


def test_sample_sigmas_log_uniform() -> None:
    from tac.self_augmentation_v2 import HighSigmaSampler, HighSigmaStrategyConfig

    cfg = HighSigmaStrategyConfig(redraw_fraction=0.0)
    gen = torch.Generator().manual_seed(1234)
    sigmas = HighSigmaSampler(cfg, generator=gen).sample_sigmas(10_000, torch.device("cpu"))

    log_min = math.log(cfg.normal_sigma_min)
    log_max = math.log(cfg.normal_sigma_max)
    normalized = (sigmas.log() - log_min) / (log_max - log_min)
    p_value = _ks_uniform_pvalue(normalized)

    assert p_value > 0.05


def test_apply_sigma_noise_is_differentiable() -> None:
    from tac.self_augmentation_v2 import apply_sigma_noise_to_input

    x = torch.zeros(4, 3, 8, 8, requires_grad=True)
    sigmas = torch.full((4,), 0.25)

    out = apply_sigma_noise_to_input(x, sigmas)
    out.sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad).item() == x.numel()


def test_apply_sigma_noise_per_sample_independent() -> None:
    from tac.self_augmentation_v2 import apply_sigma_noise_to_input

    torch.manual_seed(2026)
    x = torch.zeros(2, 1, 12, 12)
    sigmas = torch.ones(2)

    out = apply_sigma_noise_to_input(x, sigmas)
    noise = out - x

    assert not torch.equal(noise[0], noise[1])


def test_disabled_returns_zero_noise() -> None:
    from tac.self_augmentation_v2 import HighSigmaSampler, HighSigmaStrategyConfig, apply_sigma_noise_to_input

    x = torch.randn(3, 2, 6, 6)
    cfg = HighSigmaStrategyConfig(enabled=False)
    sigmas = HighSigmaSampler(cfg, generator=torch.Generator().manual_seed(9)).sample_sigmas(
        x.shape[0], x.device,
    )

    out = apply_sigma_noise_to_input(x, sigmas)

    assert torch.equal(sigmas, torch.zeros_like(sigmas))
    assert torch.equal(out, x)


def test_train_loop_integration() -> None:
    from tac.experiments.train_renderer import parse_args
    from tac.self_augmentation_v2 import HighSigmaSampler, HighSigmaStrategyConfig, apply_sigma_noise_to_input

    args = parse_args([
        "--profile", "saug_v2_dilated_h64",
        "--tag", "unit",
        "--no-auth-eval-on-best",
    ])
    assert args.use_saug_v2 is True

    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 3, kernel_size=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    torch.manual_seed(7)
    model = TinyRenderer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(4, 3, 16, 16)
    target = torch.zeros_like(x)

    cfg = HighSigmaStrategyConfig(
        redraw_fraction=args.saug_v2_redraw_fraction,
        high_sigma_min=args.saug_v2_high_sigma_min,
        high_sigma_max=args.saug_v2_high_sigma_max,
        normal_sigma_min=args.saug_v2_normal_sigma_min,
        normal_sigma_max=args.saug_v2_normal_sigma_max,
        enabled=args.use_saug_v2,
    )
    sampler = HighSigmaSampler(cfg, generator=torch.Generator().manual_seed(args.seed))
    sigmas = sampler.sample_sigmas(x.shape[0], x.device) / 255.0
    rendered = model(apply_sigma_noise_to_input(x, sigmas))
    loss = (rendered - target).pow(2).mean()

    assert math.isfinite(loss.item())
    loss.backward()
    opt.step()

    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_orthogonal_to_saug() -> None:
    from tac.self_augmentation_v2 import HighSigmaSampler, HighSigmaStrategyConfig, apply_sigma_noise_to_input

    class MockSaug:
        def perturb(self, x: torch.Tensor) -> torch.Tensor:
            return x + 0.01

    x = torch.zeros(2, 3, 8, 8, requires_grad=True)
    saug = MockSaug()
    cfg = HighSigmaStrategyConfig(redraw_fraction=0.5)
    sampler = HighSigmaSampler(cfg, generator=torch.Generator().manual_seed(55))

    sigmas = sampler.sample_sigmas(x.shape[0], x.device) / 255.0
    out = apply_sigma_noise_to_input(saug.perturb(x), sigmas)
    loss = out.square().mean()
    loss.backward()

    assert math.isfinite(loss.item())
    assert x.grad is not None
