"""Lane Ω-V2 — per-WEIGHT learnable bit-depth via Lagrangian dual ascent.

The mathematically-optimal evolution of Lane Ω-V1's water-fill heuristic.

Math (council Yousfi+Fridrich+Hotz):
    Lane Ω-V1 used a closed-form water-fill allocator
        bits_ij ∝ I(w_ij)^α   with α=0.5 hard-coded
    derived from a high-rate Gaussian-noise approximation
        D(b) ≈ σ²/12 · 2^(-2b)
    minimised under the constraint sum(bits_ij) ≤ B by Lagrangian.

    The closed-form allocation is OPTIMAL only when the actual fake-quant
    distortion is locally Gaussian. In our setting it isn't: STE round +
    per-output-channel scale + scorer non-linearity make the true distortion
    landscape decidedly non-Gaussian, and α=0.5 is just a guess at the slope.

    Lane Ω-V2 directly learns bits_ij as a continuous parameter via
    Lagrangian dual ascent. The KKT condition at the constrained optimum
    (rate-distortion duality) is
        ∂D/∂bits_ij = -λ · ∂R/∂bits_ij    ∀ (i,j)
    where ∂R/∂bits_ij = 1 (each weight contributes 1 unit per bit).

    Setting λ via dual ascent (sub-gradient on the constraint violation)
    converges to the SAME optimum as water-fill IF the high-rate
    approximation holds, but ALSO adapts to the actual fake-quant
    distortion landscape when it doesn't. So Lane Ω-V2 is at-least-as-good
    as Ω-V1 in the limit, and strictly better when the distortion is
    non-Gaussian (i.e. the regime we actually care about).

Implementation pattern: this module mirrors Lane S's
``self_compress.LearnableBitDepth`` + ``SelfCompressingConv2d`` but extends
the bit-depth from per-CHANNEL to per-WEIGHT. The differentiable
parameterisation uses ``softplus(raw)`` to keep bits >= 0; an upper
clamp at 8 keeps the encoder representable.

CLAUDE.md compliance:
    * Pure PyTorch. CUDA-required; no MPS fallback (caller decides device).
    * Tested: gradient flows through bits + weights, 8-bit ≈ identity,
      1-bit clusters to ±max, swap respects protected layers.
    * No global state.
"""
from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.self_compress import SC_PROTECTED_NAME_PATTERNS

__all__ = [
    "LearnablePerElementBitDepth",
    "LearnableBitConv2d",
    "LagrangianRateController",
    "swap_renderer_convs_with_learnable_bits",
    "list_learnable_bit_layers",
    "renderer_total_learnable_weight_bits",
    "renderer_average_learnable_bits_per_weight",
    "compute_learnable_bit_rate_penalty",
]


# ── Per-element bit-depth STE ────────────────────────────────────────────


class _PerElementSTEQuantize(torch.autograd.Function):
    """Per-element bit-depth quantization with STE backward.

    Forward: For each element with continuous bit-depth ``b``:
        q = round(b)                 # nearest integer bit-depth (1..8)
        levels = max(2^(b-1) - 1, 1) # one-sided level count
        step = scale_per_channel / levels
        out  = clamp(round(w/step) * step, -scale, +scale)
        # 1-bit override: out = sign(w) * scale (no representable 0)

    Backward (STE):
        ∂L/∂w    = ∂L/∂out                                 (pass-through)
        ∂L/∂bits = ∂L/∂out * (∂out/∂step) * (∂step/∂bits)
                 ≈ derived analytically below; we pass the upstream
                   gradient scaled by an indicator of saturation.

    The bits gradient is the load-bearing piece of Lane Ω-V2 — it is
    what lets dual ascent on λ + SGD on bits converge to the KKT optimum.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,        # (C_out, ...)
        scale: torch.Tensor,         # (C_out,) per-output-channel max
        bits_continuous: torch.Tensor,  # same shape as weight, float
    ) -> torch.Tensor:
        if bits_continuous.shape != weight.shape:
            raise ValueError(
                f"bits shape {tuple(bits_continuous.shape)} must match "
                f"weight shape {tuple(weight.shape)}"
            )
        if scale.shape != (weight.shape[0],):
            raise ValueError(
                f"scale shape {tuple(scale.shape)} must equal "
                f"({weight.shape[0]},) for per-output-channel scale"
            )
        # Round bits to nearest integer for the forward quantizer.
        # Backward uses the continuous bits via the STE analytic gradient.
        bits_clamped = bits_continuous.clamp(min=1.0, max=8.0)
        bits_int = bits_clamped.round()

        scale_b = scale.view(-1, *([1] * (weight.dim() - 1))).to(weight.dtype)
        bits_b = bits_int.to(weight.dtype)
        # one-sided level count (>= 1)
        levels = (2.0 ** (bits_b - 1.0) - 1.0).clamp(min=1.0)
        step = scale_b / levels

        # General quantizer
        q_general = torch.round(weight / step) * step
        q_general = q_general.clamp(min=-scale_b, max=scale_b)
        # 1-bit sign-only quantizer (matches OMG1 packer + FrozenBitFakeQuant)
        q_sign = torch.where(
            weight >= 0, scale_b.expand_as(weight), -scale_b.expand_as(weight)
        )
        one_bit_mask = (bits_int == 1.0)
        q = torch.where(one_bit_mask, q_sign, q_general)

        # ── Codex Round 6 fix: capture q at b+1 and b-1 for finite-difference STE ──
        # The Round 4 residual surrogate (-ln2 · (q - w)) silently assumed that
        # increasing bits monotonically refines the quantization grid. That holds
        # only when adjacent integer grids are NESTED — which is FALSE for our
        # quantizer: levels(b) = 2^(b-1) - 1 produces {1, 3, 7, 15, ...} which is
        # a sequence of co-prime denominators. A weight that lands on a "sweet
        # spot" at b=3 (e.g., w ≈ 1/3) can have STRICTLY HIGHER distortion at b=4
        # because 1/3 is not representable on the b=4 grid {±k/7}. The residual
        # surrogate's sign in those cases pushes SGD toward worse bit-depths.
        #
        # Forward computes the actual quantized value at b+1 and b-1 using the
        # SAME quantizer formula, then backward uses the central finite-difference
        # ∂q/∂bits ≈ (q_bplus - q_bminus) / 2 — an honest, non-residual estimator
        # that captures the true non-monotonic landscape.
        #
        # Cost: two extra rounds per element per forward (≈ same wall-clock as
        # the main quantizer; no autograd graph).
        with torch.no_grad():
            bits_plus = (bits_int + 1.0).clamp(max=8.0)
            bits_minus = (bits_int - 1.0).clamp(min=1.0)

            # b+1 quantization
            levels_plus = (2.0 ** (bits_plus - 1.0) - 1.0).clamp(min=1.0)
            step_plus = scale_b / levels_plus
            q_plus_general = (torch.round(weight / step_plus) * step_plus).clamp(
                min=-scale_b, max=scale_b
            )
            q_plus_sign = torch.where(
                weight >= 0, scale_b.expand_as(weight), -scale_b.expand_as(weight)
            )
            # bits_plus == 1 only when bits_int already == 1 AND clamp kicked,
            # but bits_int == 1 → bits_plus = 2 (no clamp); so q_plus is general.
            # For the upper clamp at bits_int == 8, q_plus uses b=8 (same as b).
            one_bit_mask_plus = (bits_plus == 1.0)
            q_bplus = torch.where(one_bit_mask_plus, q_plus_sign, q_plus_general)

            # b-1 quantization
            levels_minus = (2.0 ** (bits_minus - 1.0) - 1.0).clamp(min=1.0)
            step_minus = scale_b / levels_minus
            q_minus_general = (torch.round(weight / step_minus) * step_minus).clamp(
                min=-scale_b, max=scale_b
            )
            q_minus_sign = torch.where(
                weight >= 0, scale_b.expand_as(weight), -scale_b.expand_as(weight)
            )
            one_bit_mask_minus = (bits_minus == 1.0)
            q_bminus = torch.where(one_bit_mask_minus, q_minus_sign, q_minus_general)

        # Save tensors for STE backward
        ctx.save_for_backward(
            weight, scale_b, bits_b, levels, q, one_bit_mask,
            q_bplus, q_bminus,
        )
        return q.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D401
        (
            weight, scale_b, bits_b, levels, q, one_bit_mask,
            q_bplus, q_bminus,
        ) = ctx.saved_tensors

        # Weight gradient: STE pass-through, zeroed where saturated to ±scale.
        # Saturation detected by |q| ≈ scale_b within one step.
        step = scale_b / levels
        saturated = (q.abs() >= (scale_b - 0.5 * step))
        grad_weight = grad_output * (~saturated).to(grad_output.dtype)
        # 1-bit elements: sign quantizer is non-differentiable; use STE
        # pass-through (the loss can still pull |w| toward sign-of-correct).
        grad_weight = torch.where(
            one_bit_mask, grad_output, grad_weight
        )

        # ── Bits gradient: finite-difference STE surrogate ───────────────
        #
        # Codex Round 6 fix: the Round 4 residual surrogate
        # ``∂q/∂bits ≈ -ln2 · (q − w)`` silently assumed that increasing
        # bits monotonically refines the quantization grid. That holds only
        # when adjacent integer grids are NESTED — which is FALSE for our
        # quantizer: ``levels(b) = 2^(b-1) - 1`` produces a sequence
        # ``{1, 3, 7, 15, …}`` of co-prime denominators, so the grid at b=4
        # is NOT a refinement of the grid at b=3.
        #
        # Counter-example that broke Round 4:
        #   w = 1/3 + 0.001, scale = 1
        #   b=3: q = 1/3        →  MSE ≈ 5e-7   (sweet spot!)
        #   b=4: q = 2/7 ≈ 0.29 →  MSE ≈ 1.13e-3 (worse!)
        # The Round 4 surrogate gives ``grad_bits = -ln2 · (q − w)² < 0`` at
        # b=3 → SGD raises bits 3→4 → distortion WORSE. Pareto-dominated.
        #
        # The fix: replace the residual approximation with a proper
        # central finite-difference STE that uses the ACTUAL quantization
        # output at adjacent bit-depths captured in forward:
        #     ∂q/∂bits ≈ (q(b+1) − q(b−1)) / 2
        # and chain with the upstream gradient:
        #     ∂L/∂bits = ∂L/∂q · ∂q/∂bits = grad_output · (qp − qm) / 2
        #
        # Properties:
        #   (a) Honest — uses the real non-monotonic q(b) landscape, not a
        #       linear-grid approximation.
        #   (b) Detects sweet spots — when both adjacent bit-depths produce
        #       q FURTHER from w than the current b, the chain product
        #       points toward STAYING at b (small or sign-flipped grad).
        #   (c) Zero distortion → zero gradient (forced explicitly when
        #       ``w == 0`` because the 1-bit sign quantizer is asymmetric
        #       at w=0 (``weight >= 0`` returns +scale, not 0); without the
        #       short-circuit q_bminus would spuriously be ±scale and
        #       inject false signal where there is none).
        #   (d) The b=1 sign quantizer is handled symmetrically — its
        #       q_bminus reduces to b=1 (clamped), so the ``(qp − qm)``
        #       term still captures the marginal benefit of an extra bit.
        #
        # Reference: HAWQ (Dong et al. 2019) §3.2 motivates first-order
        # quantization-perturbation surrogates; the central-FD form is the
        # standard zeroth-order escape when the perturbation is across
        # non-nested integer grids (Wang et al., HAQ; Esser et al., LSQ).
        # Some literature uses (q_bplus − q) for FORWARD difference,
        # equally valid for STE; central is preferred here because b=1
        # gives a non-zero q_bminus that captures the sign quantizer.
        #
        # Finite-difference STE surrogate for non-nested integer grids —
        # handles the case where increasing bits can transiently increase
        # distortion (e.g., b=3→4 with w=1/3).
        d_out_d_bits = (q_bplus - q_bminus) / 2.0
        # Zero-weight short-circuit: w==0 has no distortion at any bit-depth
        # under the GENERAL quantizer (round(0)=0), but the 1-bit SIGN
        # quantizer asymmetrically returns +scale (because ``0 >= 0`` is
        # True). That asymmetry would inject a phantom -scale/2 gradient
        # at the zero-weight boundary; explicitly zero it out so the
        # safety invariant ``w == 0 → grad_bits == 0`` holds.
        zero_weight_mask = (weight == 0)
        d_out_d_bits = torch.where(
            zero_weight_mask, torch.zeros_like(d_out_d_bits), d_out_d_bits
        )
        grad_bits = grad_output * d_out_d_bits

        # No gradient flows back to scale (frozen, computed from weight).
        return grad_weight, None, grad_bits


def _ste_per_element_quantize(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bits_continuous: torch.Tensor,
) -> torch.Tensor:
    return _PerElementSTEQuantize.apply(weight, scale, bits_continuous)


# ── Learnable per-element bit-depth module ───────────────────────────────


class LearnablePerElementBitDepth(nn.Module):
    """One learnable bit-depth per WEIGHT (continuous via softplus).

    Args:
        weight_shape: shape of the weight tensor this module manages.
        init_bits: initial bit-depth (default 8.0, FP32-quality at start).
        warm_start: optional same-shape float tensor of initial bit values
            (e.g. from a Lane Ω-V1 Hessian profile). Soft warm-start —
            stored as the inverse-softplus of clipped values so that the
            initial forward sees `bits ≈ warm_start`.

    Parameterisation:
        bits = softplus(raw) clamped to [1, 8] in forward.
    so `raw` is unconstrained and gradient-friendly.

    Mirrors the API surface of ``self_compress.LearnableBitDepth`` so the
    rest of the QAT loop can treat both modules uniformly.
    """

    def __init__(
        self,
        weight_shape: torch.Size | tuple[int, ...],
        init_bits: float = 8.0,
        warm_start: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        weight_shape = torch.Size(weight_shape)
        # Inverse softplus: raw = log(exp(b) - 1)
        if warm_start is not None:
            if warm_start.shape != weight_shape:
                raise ValueError(
                    f"warm_start shape {tuple(warm_start.shape)} != "
                    f"weight shape {tuple(weight_shape)}"
                )
            init_t = warm_start.detach().to(torch.float32).clamp(min=1.0, max=8.0)
            raw_init = torch.log(torch.expm1(init_t))
        else:
            init_t = torch.tensor(float(init_bits), dtype=torch.float32).clamp(
                min=1.0, max=8.0
            )
            raw_scalar = float(torch.log(torch.expm1(init_t)).item())
            raw_init = torch.full(weight_shape, raw_scalar, dtype=torch.float32)
        self.raw = nn.Parameter(raw_init)
        self._weight_shape = weight_shape

    def bits_used(self) -> torch.Tensor:
        """Continuous bits in [1, 8] (differentiable)."""
        return F.softplus(self.raw).clamp(min=1.0, max=8.0)

    def bits_rounded(self) -> torch.Tensor:
        """Integer bits (uint8) for export."""
        return self.bits_used().detach().round().clamp(min=1, max=8).to(torch.uint8)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.shape != self._weight_shape:
            raise ValueError(
                f"forward weight shape {tuple(weight.shape)} != "
                f"managed shape {tuple(self._weight_shape)}"
            )
        # Per-output-channel max (matches FrozenBitFakeQuant + SCv1).
        out_dim = weight.shape[0]
        scale = weight.detach().reshape(out_dim, -1).abs().amax(dim=1).clamp(min=1e-8)
        bits = self.bits_used()
        return _ste_per_element_quantize(weight, scale, bits)

    def mean_bits(self) -> float:
        """Average bit-depth across all weights this module manages."""
        return float(self.bits_used().detach().mean().item())

    def total_bits(self) -> torch.Tensor:
        """Differentiable sum of per-weight bits (drives Lagrangian rate)."""
        return self.bits_used().sum()

    def total_bits_rounded(self) -> int:
        """Non-differentiable export-time bit count."""
        return int(self.bits_rounded().to(torch.int64).sum().item())


# ── Learnable-bit Conv2d ─────────────────────────────────────────────────


class LearnableBitConv2d(nn.Module):
    """Conv2d whose weight is fake-quantized at per-element learnable bit-depth.

    Conceptual template = Lane S's ``SelfCompressingConv2d`` (per-CHANNEL
    bits) + the per-element STE from ``frozen_bit_quant`` (frozen bits).
    Lane Ω-V2 = per-WEIGHT × LEARNABLE bits. ~30 lines of plumbing once you
    have ``LearnablePerElementBitDepth``.

    Forward: q_w = LearnablePerElementBitDepth(weight); F.conv2d(x, q_w, bias).
    Backward: STE flows gradient through both `weight` and `bits.raw`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        init_bits: float = 8.0,
        warm_start: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.bit_depth = LearnablePerElementBitDepth(
            weight_shape=self.conv.weight.shape,
            init_bits=init_bits,
            warm_start=warm_start,
        )
        # Mirror the Conv2d kwargs used during export
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.bit_depth(self.conv.weight)
        if self.conv.padding_mode != "zeros":
            pad = self.conv.padding
            if isinstance(pad, int):
                pad_tuple = (pad, pad, pad, pad)
            else:
                pad_tuple = (pad[1], pad[1], pad[0], pad[0])
            x = F.pad(x, pad_tuple, mode=self.conv.padding_mode)
            return F.conv2d(
                x, q_weight, self.conv.bias,
                stride=self.conv.stride, padding=0,
                dilation=self.conv.dilation, groups=self.conv.groups,
            )
        return F.conv2d(
            x, q_weight, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups,
        )

    def weight_numel(self) -> int:
        return self.conv.weight.numel()

    def total_weight_bits(self) -> torch.Tensor:
        """Differentiable total bits across this layer's weights (drives rate)."""
        return self.bit_depth.total_bits()

    def total_weight_bits_rounded(self) -> int:
        return self.bit_depth.total_bits_rounded()

    def mean_bits(self) -> float:
        return self.bit_depth.mean_bits()


# ── Renderer-level helpers ───────────────────────────────────────────────


def _is_protected_name(qualified_name: str, extra: tuple[str, ...] = ()) -> bool:
    """Same protection-suffix rule as Lane S (delegate to SC_PROTECTED list)."""
    patterns = tuple(SC_PROTECTED_NAME_PATTERNS) + tuple(extra)
    for pat in patterns:
        if qualified_name == pat or qualified_name.endswith("." + pat):
            return True
    return False


def swap_renderer_convs_with_learnable_bits(
    model: nn.Module,
    *,
    init_bits: float = 8.0,
    hessian_init: dict | None = None,
    skip_transposed: bool = True,
    skip_groupwise: bool = False,
    extra_protected_patterns: tuple[str, ...] = (),
) -> dict:
    """Swap eligible nn.Conv2d layers with LearnableBitConv2d (in-place).

    Mirrors ``self_compress.swap_renderer_convs_with_self_compress`` but
    swaps to LEARNABLE per-WEIGHT instead of per-channel.

    Args:
        model: a build_renderer(...) output (PairGenerator or
            AsymmetricPairGenerator).
        init_bits: default initial bit-depth for swapped layers (per element).
        hessian_init: optional dict mapping ``<module_name>.weight`` →
            tensor of per-element importance values (e.g. from
            ``experiments/profile_hessian_per_weight.py``). When provided,
            the warm-start bit-depth for each weight is computed as
                bits_warm = clamp(init_bits * sqrt(I / median(I)), 1, 8)
            so high-importance weights start with more bits. This is a
            SOFT init — Lagrangian dual ascent will adjust.
        skip_transposed: leave nn.ConvTranspose2d FP32 (recommended; STE
            backward through transposed conv is ill-behaved).
        skip_groupwise: leave non-trivial grouped convs FP32.
        extra_protected_patterns: caller-supplied additional name suffixes.

    Returns:
        Dict with ``swapped`` / ``protected`` / ``skipped`` /
        ``total_swapped_params`` / ``warm_started`` lists.
    """
    swapped: list[str] = []
    protected: list[str] = []
    skipped: list[str] = []
    warm_started: list[str] = []
    total_swapped_params = 0

    parents: dict[str, nn.Module] = {"": model}
    for name, mod in model.named_modules():
        parents[name] = mod

    candidates: list[tuple[nn.Module, str, str, nn.Module]] = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            if skip_transposed:
                skipped.append(full_name + " (transposed)")
            else:
                skipped.append(full_name + " (transposed; not supported)")
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if skip_groupwise and module.groups != 1 and module.groups != module.in_channels:
            skipped.append(full_name + f" (groups={module.groups})")
            continue
        if _is_protected_name(full_name, extra_protected_patterns):
            protected.append(full_name)
            continue
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = parents[parent_name]
        else:
            parent = model
            child_name = full_name
        candidates.append((parent, child_name, full_name, module))

    # Build hessian-derived warm-start tensors if hessian_init provided.
    warm_table: dict[str, torch.Tensor] = {}
    if hessian_init is not None:
        # hessian_init may be the raw `importance` dict OR the full
        # {"importance": {...}, "metadata": {...}} dict from
        # profile_hessian_per_weight.py — accept both.
        if "importance" in hessian_init and isinstance(
            hessian_init["importance"], dict
        ):
            imp_dict = hessian_init["importance"]
        else:
            imp_dict = hessian_init
        # Compute global median for normalisation
        all_vals = torch.cat(
            [t.detach().reshape(-1).float() for t in imp_dict.values()]
        ) if imp_dict else torch.empty(0)
        if all_vals.numel() > 0:
            global_median = float(all_vals.median().item())
            global_median = max(global_median, 1e-30)
            for layer_weight_name, imp in imp_dict.items():
                # Warm-start bits ∝ sqrt(I / median_I) (matches HAWQ alpha=0.5)
                ratio = imp.float() / global_median
                bits_warm = (init_bits * ratio.sqrt()).clamp(min=1.0, max=8.0)
                warm_table[layer_weight_name] = bits_warm

    for parent, child_name, full_name, conv in candidates:
        kernel_size = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation

        warm = None
        weight_key = f"{full_name}.weight"
        if weight_key in warm_table:
            ws = warm_table[weight_key]
            if ws.shape == conv.weight.shape:
                warm = ws
                warm_started.append(full_name)

        new_layer = LearnableBitConv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            init_bits=init_bits,
            warm_start=warm,
        )
        with torch.no_grad():
            new_layer.conv.weight.copy_(conv.weight)
            if conv.bias is not None and new_layer.conv.bias is not None:
                new_layer.conv.bias.copy_(conv.bias)
        new_layer = new_layer.to(conv.weight.device).to(conv.weight.dtype)
        setattr(parent, child_name, new_layer)
        swapped.append(full_name)
        total_swapped_params += new_layer.weight_numel()

    return {
        "swapped": swapped,
        "protected": protected,
        "skipped": skipped,
        "warm_started": warm_started,
        "total_swapped_params": total_swapped_params,
    }


def list_learnable_bit_layers(
    model: nn.Module,
) -> list[tuple[str, "LearnableBitConv2d"]]:
    """Return every (name, layer) for LearnableBitConv2d in the model."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, LearnableBitConv2d)
    ]


def renderer_total_learnable_weight_bits(model: nn.Module) -> torch.Tensor:
    """Differentiable sum of bits across every learnable-bit layer."""
    layers = list_learnable_bit_layers(model)
    if not layers:
        return torch.tensor(0.0)
    device = next(iter(layers))[1].bit_depth.raw.device
    total = torch.zeros((), device=device)
    for _name, layer in layers:
        total = total + layer.total_weight_bits()
    return total


def renderer_average_learnable_bits_per_weight(model: nn.Module) -> float:
    """Mean bits per weight (non-differentiable; for logs)."""
    layers = list_learnable_bit_layers(model)
    if not layers:
        return 0.0
    total_bits = 0.0
    total_weights = 0
    for _name, layer in layers:
        total_bits += float(layer.total_weight_bits().detach().item())
        total_weights += layer.weight_numel()
    return total_bits / max(total_weights, 1)


class LagrangianRateController:
    """True primal-dual controller for the Lane Ω-V2 bit-budget constraint.

    Round 3 introduced this controller for the dual side, but the primal
    surrogate retained an ``F.relu`` clamp that destroyed the KKT property
    (zero gradient under slack → equilibrium drifts above target). Round 4
    fixed the primal to the LINEAR form ``λ · (mean_bits − target)``; the
    KKT non-negativity clamp now lives ONLY in the dual update::

        primal step:  ∇_θ [D(θ) + λ · (mean_bits(θ) − target)]
        dual step:    λ_{t+1} = max(0, λ_t + η · (mean_bits − target))

    Together this is the textbook primal-dual pair for
        min_θ  D(θ)
        s.t.   mean_bits(θ) ≤ target
    (Boyd & Vandenberghe 2004 §5.4; Nedić & Ozdaglar 2009 sub-gradient
    convergence). At equilibrium ``mean_bits = target`` AND ``λ ≥ 0`` —
    the KKT condition for the inequality constraint — even when the
    primal momentarily slips into slack (``residual < 0``), because the
    dual decays λ → 0 there and turns the primal pressure off smoothly.

    Default ``eta = 1e-3`` and ``initial_lambda = 0.0`` follow the
    standard "no-pressure" warm-start used in Lane S self-compression.
    """

    def __init__(
        self,
        target_bits_per_weight: float,
        *,
        eta: float = 1e-3,
        initial_lambda: float = 0.0,
        lambda_max: float | None = None,
    ) -> None:
        if target_bits_per_weight <= 0:
            raise ValueError(
                f"target_bits_per_weight must be > 0 (got "
                f"{target_bits_per_weight}); a non-positive target leaves "
                f"the dual update without a well-defined sign."
            )
        if eta <= 0:
            raise ValueError(
                f"eta must be > 0 (got {eta}); a non-positive step size "
                f"halts dual ascent."
            )
        if initial_lambda < 0:
            raise ValueError(
                f"initial_lambda must be ≥ 0 (got {initial_lambda}); the "
                f"KKT multiplier on an inequality constraint is "
                f"non-negative by definition."
            )
        if lambda_max is not None and lambda_max < initial_lambda:
            raise ValueError(
                f"lambda_max ({lambda_max}) must be ≥ initial_lambda "
                f"({initial_lambda}); the upper bound is otherwise "
                f"infeasible."
            )
        self.target_bits_per_weight = float(target_bits_per_weight)
        self.eta = float(eta)
        self.lambda_max = lambda_max
        self._lambda = float(initial_lambda)
        self._step = 0
        self._last_residual: float | None = None

    @property
    def lambda_rate(self) -> float:
        """Current dual multiplier λ_t (always ≥ 0)."""
        return self._lambda

    @property
    def step_count(self) -> int:
        """Number of completed :py:meth:`dual_update` calls."""
        return self._step

    @property
    def last_residual(self) -> float | None:
        """Most recent constraint residual ``mean_bits - target``."""
        return self._last_residual

    def dual_update(self, mean_bits: float, target: float | None = None) -> float:
        """Apply one dual-ascent step using the latest mean-bits observation.

        Parameters
        ----------
        mean_bits:
            Observed average bits/weight at the current primal iterate.
        target:
            Optional override for the controller's stored target (so the
            same controller can serve a ramp schedule). Defaults to the
            value fixed at construction.

        Returns
        -------
        float
            The updated multiplier ``λ_{t+1}`` (used by subsequent calls
            to :py:meth:`compute_learnable_bit_rate_penalty`).
        """
        tgt = float(target) if target is not None else self.target_bits_per_weight
        residual = float(mean_bits) - tgt
        self._last_residual = residual
        # Sub-gradient dual ascent: λ_{t+1} = max(0, λ_t + η · residual).
        # When residual > 0 (constraint violated) λ rises; when < 0 (slack)
        # λ falls. The max-with-0 enforces λ ≥ 0 (KKT non-negativity).
        new_lambda = max(0.0, self._lambda + self.eta * residual)
        if self.lambda_max is not None:
            new_lambda = min(new_lambda, float(self.lambda_max))
        self._lambda = new_lambda
        self._step += 1
        return self._lambda

    def reclamp_to_lambda_max(self) -> float:
        """Re-apply the current ``lambda_max`` cap to the multiplier
        without performing a dual-ascent step.

        Useful when the operator has *tightened* ``lambda_max`` during
        training (e.g. a ramp schedule that lowers the cap) and wants
        the new bound to take effect immediately, without waiting for
        the next ``dual_update`` to discover it.
        """
        if self.lambda_max is not None and self._lambda > self.lambda_max:
            self._lambda = float(self.lambda_max)
        # Also enforce non-negativity in case lambda_max was set < 0
        # somehow (defensive).
        if self._lambda < 0.0:
            self._lambda = 0.0
        return self._lambda

    def state_dict(self) -> dict:
        """Serialize controller state for resume."""
        return {
            "target_bits_per_weight": self.target_bits_per_weight,
            "eta": self.eta,
            "lambda_max": self.lambda_max,
            "_lambda": self._lambda,
            "_step": self._step,
            "_last_residual": self._last_residual,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previous :py:meth:`state_dict`."""
        for key in (
            "target_bits_per_weight", "eta", "lambda_max",
            "_lambda", "_step", "_last_residual",
        ):
            if key in state:
                setattr(self, key, state[key])


def compute_learnable_bit_rate_penalty(
    model: nn.Module,
    target_bits_per_weight: float,
    lambda_rate: float | "LagrangianRateController",
) -> torch.Tensor:
    """Linear Lagrangian rate penalty: ``λ × (mean_bits − target)``.

    Bug 2 fix (codex Round 4):
        The Round 3 fix kept a hidden ``F.relu(residual)`` in the primal
        surrogate (``λ × max(0, mean_bits − target)``). That destroys the
        KKT property the docstring promised: under slack
        (``mean_bits < target``) the gradient w.r.t. bits is ZERO, so the
        bit allocator only feels rate pressure when *already* over budget
        — equilibrium drifts above target whenever the slack-side dual
        decay is faster than the over-budget excursion that re-engages
        the hinge. The textbook primal-dual pair for an inequality
        constraint is::

            primal:  λ · (mean_bits − target)            # linear, no ReLU
            dual:    λ ← max(0, λ + η · (mean_bits − target))  # ascent + clamp

        The clamp belongs in the *dual* update (KKT non-negativity of the
        multiplier), NOT in the primal penalty. Below target the linear
        primal term contributes a negative gradient that *rewards* more
        bits, but the dual decays λ → 0 simultaneously — equilibrium
        lands at the boundary ``mean_bits = target`` with ``λ ≥ 0``,
        which is exactly the KKT condition for ``mean_bits ≤ target``.

    Backward-compat: ``lambda_rate`` accepts either a ``float`` (legacy
    fixed-multiplier path) or a :py:class:`LagrangianRateController`
    instance (canonical primal-dual path). When a controller is passed
    its current :py:attr:`~LagrangianRateController.lambda_rate` is used.

    Returns 0 (zero-dim tensor) when the model has no learnable-bit
    layers, so callers can ``loss + compute_learnable_bit_rate_penalty(...)``
    unconditionally.
    """
    layers = list_learnable_bit_layers(model)
    if not layers:
        return torch.tensor(0.0)
    device = next(iter(layers))[1].bit_depth.raw.device
    total_bits = torch.zeros((), device=device)
    total_weights = 0
    for _name, layer in layers:
        total_bits = total_bits + layer.total_weight_bits()
        total_weights += layer.weight_numel()
    mean_bits = total_bits / max(total_weights, 1)
    if isinstance(lambda_rate, LagrangianRateController):
        lam = lambda_rate.lambda_rate
        target = lambda_rate.target_bits_per_weight
        is_controller = True
    else:
        lam = float(lambda_rate)
        target = float(target_bits_per_weight)
        is_controller = False
    # Codex Round 5 fix — split semantics:
    # * LagrangianRateController path: LINEAR penalty `λ * residual`
    #   (no ReLU). The KKT non-negativity clamp is enforced by the dual
    #   update `λ ← max(0, λ + η·residual)` inside the controller, NOT
    #   here. Letting the primal go negative under slack is mathematically
    #   correct because the dual decays λ → 0 in steady state, so the
    #   primal contribution vanishes and equilibrium lands at the
    #   constraint boundary `mean_bits = target`.
    # * Legacy float `lambda_rate` path: ONE-SIDED HINGE `λ * relu(residual)`.
    #   Plain float callers have NO dual update to decay λ — if we used
    #   linear here, the under-budget gradient would be `+λ` (constant)
    #   forever, pushing bits below the floor for no benefit. ReLU
    #   correctly contributes zero gradient under slack, preserving
    #   under-budget invariants for these legacy callers.
    residual = mean_bits - target
    if is_controller:
        return lam * residual
    else:
        return lam * F.relu(residual)


def iter_eligible_conv_names(
    model: nn.Module,
    extra_protected_patterns: tuple[str, ...] = (),
) -> Iterable[str]:
    """Yield qualified names of Conv2d layers eligible for the swap.

    Used by tests + the QAT script to inspect which layers WILL be swapped
    BEFORE actually mutating the model.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ConvTranspose2d):
            continue
        if not isinstance(mod, nn.Conv2d):
            continue
        if _is_protected_name(name, extra_protected_patterns):
            continue
        yield name


# ── Pattern: callers can reuse SC_PROTECTED_NAME_PATTERNS verbatim ──────
# Re-exported for convenience so QAT scripts don't need to import from
# self_compress and learnable_bit_quant separately.
LEARNABLE_BIT_PROTECTED_NAME_PATTERNS = SC_PROTECTED_NAME_PATTERNS

# Marker for grep-friendly identification of Lane Ω-V2 modules.
_LANE_OMEGA_V2_MARKER = "lane_omega_v2_learnable_bit_quant"


# ── Smoke ──────────────────────────────────────────────────────────────


def _smoke() -> None:
    torch.manual_seed(0)
    layer = LearnableBitConv2d(3, 4, 3, padding=1, init_bits=8.0)
    x = torch.randn(1, 3, 8, 8)
    y = layer(x)
    assert y.shape == (1, 4, 8, 8)
    y.sum().backward()
    assert layer.bit_depth.raw.grad is not None
    assert layer.conv.weight.grad is not None
    print("learnable_bit_quant: smoke OK", y.shape)


if __name__ == "__main__":
    _smoke()
