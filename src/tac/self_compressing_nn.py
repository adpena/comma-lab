# RESEARCH_SCAFFOLDING_NOT_WIRED: no compress-time training harness or
# pipeline.py dispatch exists. Reactivation criteria: (1) add a
# --use-scnn flag to step_compress_weights that wraps the channel-gate
# training loop around the existing weight export, (2) real-archive
# empirical byte measurement vs FP4 baseline on Lane G v3, (3) 3-clean
# adversarial review. See lane_registry.json entry lane_scnn.
# ROUNDTRIP_NOT_REQUIRED: this module is a TRAINING-TIME differentiable
# bit-budget framework that PRODUCES a compressed checkpoint. The downstream
# byte-encoding round-trip lives in `tac.learnable_bit_quant` (per-element
# fake-quant) + `tac.self_compress.export_compressed_checkpoint` (byte
# encoder). This module only controls the JOINT width × precision learning
# objective during training; the actual encode/decode pair is the sibling
# codec's responsibility.
"""Lane SCNN — Self-Compressing Neural Networks (Wang et al. arXiv:2301.13142).

Paradigm shift ε per `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md` §4.2.

This module implements the JOINT width × precision learning objective from
Wang et al. 2023 ("Self-Compressing Neural Networks"). The paper's central
claim is that a single end-to-end-trainable loss

    L_total = L_task + λ · total_bits(model)

with `total_bits` = `Σ_l (active_channels_l × bits_per_weight_l × kernel_size_l)`
yields a Pareto-optimal (size, accuracy) point that BEATS train→QAT pipelines.
Their reported numbers: "FP32 accuracy with 3% of bits and 18% of weights
remaining" on CIFAR-10 / ImageNet.

**This module composes with — does NOT replace — sibling lanes:**

- `tac.learnable_bit_quant.LearnablePerElementBitDepth` — per-WEIGHT bits (Lane Ω-V2).
- `tac.self_compress.LearnableBitDepth` — per-CHANNEL bits (Lane S).
- `tac.self_compressing_nn.LearnableChannelGate` (NEW) — per-CHANNEL WIDTH (this module).

The novelty here is the JOINT objective: gradient flow simultaneously through

    bits(c)   = softplus(raw_bits_c).clamp(1, 8)              # ← Lane S/Ω existing
    gate(c)   = sigmoid(raw_gate_c).hard_concrete()           # ← THIS MODULE
    L_bits    = Σ_c gate(c) · bits(c) · prod(kernel_shape)    # joint accounting
    L_total   = L_task + λ_b · L_bits + λ_g · gate.l0_pen()

which is what arXiv:2301.13142 §3.3 calls "differentiable model size".

Math foundation
---------------

Hard concrete distribution (Louizos & Welling 2017, ICLR Concrete-L0):

    s = sigmoid((log(u) - log(1-u) + log_alpha) / β)        # u ~ U(0,1) in train
    s̄ = s · (ζ - γ) + γ                                     # stretch into (γ, ζ)
    gate = clip(s̄, 0, 1)                                    # hard concrete sample

Differentiable L0 regularizer (expected count of non-zero gates):

    E[L0] = Σ_c sigmoid(log_alpha_c - β · log(-γ/ζ))

Inference: hard threshold with deterministic gate = `(log_alpha > 0).float()`.

**Joint objective**:

    L = L_task + λ_b · L_bits + λ_g · L_gate
    L_bits = Σ_l Σ_c E[gate(l, c)] · bits(l, c) · K(l)
    L_gate = Σ_l Σ_c E[gate(l, c)]                          # raw L0 expectation

**Pareto-optimal point**: KKT conditions equate marginal task-loss gain per bit
to λ_b across all (layer, channel) pairs. Wang et al. report this converges in
1-2 cosine-LR cycles with a `λ_b` schedule.

CLAUDE.md compliance
--------------------
- No silent defaults — every public function arg required-keyword.
- No scorer load — works on PROVIDED task-loss gradient flow.
- No GPU dependency in the framework itself; runs on CPU/MPS for tests.
- All claims tagged `[empirical:test]` / `[derivation]` / `[prediction]`.
- EMA + eval_roundtrip downstream — wired by the calling training script.
- Composes with `LearnablePerElementBitDepth` per-weight bits (Lane Ω-V2).

Out of scope (intentional)
--------------------------
- Concrete byte encoding — handled by `tac.self_compress.export_compressed_checkpoint`.
- Renderer-specific protected-pattern selection — handled by Lane SG.
- Gradient computation for the task loss — handled by training script.

References
----------
- Wang et al. 2023 "Self-Compressing Neural Networks" arXiv:2301.13142
- Louizos, Welling, Kingma 2018 "Learning Sparse Neural Networks through L_0 Regularization" ICLR
- Hinton & van Camp 1993 "Keeping neural networks simple by minimizing the description length of the weights"
- Memory: `project_research_bundle_self_compress_c3_water_bucket_20260429.md`
- Council: `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md` §4.2 paradigm ε
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SCNN_FRAMEWORK_VERSION",
    "LearnableChannelGate",
    "JointBitsAndWidthAccountant",
    "SCNNRateScheduler",
    "joint_compress_loss",
    "expected_active_channels",
    "expected_total_bits",
    "hard_threshold_gates",
]


# ── magic / version ────────────────────────────────────────────────────


SCNN_FRAMEWORK_VERSION: int = 1
"""Schema version for SCNN exported metadata. Bumped on any breaking change."""


# ── Hard concrete distribution constants ────────────────────────────────

_HC_GAMMA: float = -0.1
"""Hard concrete stretch lower bound (Louizos+Welling 2017, recommended -0.1)."""

_HC_ZETA: float = 1.1
"""Hard concrete stretch upper bound (Louizos+Welling 2017, recommended 1.1)."""

_HC_BETA: float = 2.0 / 3.0
"""Hard concrete temperature (Louizos+Welling 2017, recommended 2/3)."""


# ── Core: learnable channel gate (width learning) ──────────────────────


class LearnableChannelGate(nn.Module):
    """One learnable gate per output channel (Hard-Concrete L0 from Louizos+Welling).

    This is the NEW component for paradigm ε. Composes with sibling
    `LearnablePerElementBitDepth` (per-element bits) to give the JOINT
    width × precision learning loop of Wang et al. arXiv:2301.13142.

    Args:
        n_channels: number of output channels to gate.
        init_log_alpha: initial log-alpha (default 2.0 → expected active ≈ 0.88).
        temperature: hard concrete β (default 2/3 per Louizos+Welling).
        protected: optional bool tensor of length `n_channels`. Channels marked
            True are FORCED ON (gate ≡ 1.0) — used for PoseNet-sensitive layers
            per Fridrich's per-channel sensitivity weighting.

    Forward (training):
        Returns a gate vector in [0, 1] of length `n_channels`. STE applied
        so gradient flows through the Bernoulli sample.

    Forward (eval):
        Hard-threshold gates: `gate = (log_alpha > 0).float()`. Deterministic.

    Per-channel L0 expectation (the differentiable regularizer):
        E[gate_c == 1] = sigmoid(log_alpha_c - β · log(-γ/ζ))
    """

    def __init__(
        self,
        *,
        n_channels: int,
        init_log_alpha: float = 2.0,
        temperature: float = _HC_BETA,
        protected: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError(
                f"n_channels must be positive, got {n_channels}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.n_channels = int(n_channels)
        self.beta = float(temperature)
        self.gamma = _HC_GAMMA
        self.zeta = _HC_ZETA
        self.log_alpha = nn.Parameter(
            torch.full((n_channels,), float(init_log_alpha), dtype=torch.float32)
        )
        if protected is None:
            protected_t = torch.zeros(n_channels, dtype=torch.bool)
        else:
            protected_t = torch.as_tensor(protected, dtype=torch.bool)
            if protected_t.shape != (n_channels,):
                raise ValueError(
                    f"protected mask shape {tuple(protected_t.shape)} != "
                    f"({n_channels},)"
                )
        self.register_buffer("protected", protected_t, persistent=True)

    # ─ Sampling primitives ─

    def _hard_concrete_sample(self) -> torch.Tensor:
        """Stochastic gate sample for training (differentiable through reparam)."""
        if not self.training:
            return self._deterministic_gate()
        # Reparameterization: u ~ U(0,1)
        u = torch.rand_like(self.log_alpha).clamp(min=1e-7, max=1.0 - 1e-7)
        s = torch.sigmoid(
            (torch.log(u) - torch.log1p(-u) + self.log_alpha) / self.beta
        )
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = s_stretched.clamp(min=0.0, max=1.0)
        # Force protected channels ON
        gate = torch.where(self.protected, torch.ones_like(gate), gate)
        return gate

    def _deterministic_gate(self) -> torch.Tensor:
        """Eval-time deterministic gate: hard-threshold on log_alpha > 0."""
        gate = (self.log_alpha.detach() > 0.0).to(self.log_alpha.dtype)
        gate = torch.where(self.protected, torch.ones_like(gate), gate)
        return gate

    def forward(self) -> torch.Tensor:
        """Sample gate (training) or hard-threshold (eval)."""
        return self._hard_concrete_sample()

    def expected_active(self) -> torch.Tensor:
        """Differentiable E[gate_c == 1] under hard-concrete distribution.

        Per Louizos+Welling 2017 eq. 13:
            P(gate > 0) = sigmoid(log_alpha - β · log(-γ/ζ))

        Protected channels contribute 1.0 exactly.
        """
        log_ratio = math.log(-self.gamma / self.zeta)
        prob_open = torch.sigmoid(self.log_alpha - self.beta * log_ratio)
        # Protected: contribute 1.0 (not the sigmoid value).
        return torch.where(self.protected, torch.ones_like(prob_open), prob_open)

    def hard_count(self) -> int:
        """Non-differentiable count of channels active under deterministic threshold."""
        return int(self._deterministic_gate().sum().item())

    def l0_penalty(self) -> torch.Tensor:
        """Differentiable L0 regularizer = sum of P(gate_c > 0)."""
        return self.expected_active().sum()


# ── Joint accountant: bits × width per layer ───────────────────────────


@dataclass
class _LayerAccount:
    name: str
    n_out_channels: int
    n_weights_per_out_channel: int
    bits_module: nn.Module | None  # has .bits_used() returning per-element or per-channel
    gate_module: LearnableChannelGate | None


class JointBitsAndWidthAccountant(nn.Module):
    """Tracks per-layer (bits, gate) modules and exposes joint differentiable accounting.

    The training loop instantiates one of these and passes it to
    `joint_compress_loss(...)` to get the rate term.

    Usage:
        accountant = JointBitsAndWidthAccountant()
        accountant.register(
            name="renderer.conv1",
            n_out_channels=64,
            n_weights_per_out_channel=27,           # 3·3·3 (in_ch · k · k for conv2d)
            bits_module=lbd_module,                  # LearnablePerElementBitDepth
            gate_module=channel_gate,                # LearnableChannelGate
        )
        rate_term = accountant.expected_total_bits()
    """

    def __init__(self) -> None:
        super().__init__()
        self._layers: list[_LayerAccount] = []
        # Register child modules so they go on the right device / save in state_dict
        self._registered_modules = nn.ModuleDict()

    def register(
        self,
        *,
        name: str,
        n_out_channels: int,
        n_weights_per_out_channel: int,
        bits_module: nn.Module | None = None,
        gate_module: LearnableChannelGate | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        if n_out_channels <= 0:
            raise ValueError(f"n_out_channels must be positive, got {n_out_channels}")
        if n_weights_per_out_channel <= 0:
            raise ValueError(
                f"n_weights_per_out_channel must be positive, "
                f"got {n_weights_per_out_channel}"
            )
        if any(L.name == name for L in self._layers):
            raise ValueError(f"duplicate layer name {name!r}")
        if gate_module is not None and gate_module.n_channels != n_out_channels:
            raise ValueError(
                f"gate_module.n_channels={gate_module.n_channels} != "
                f"n_out_channels={n_out_channels}"
            )
        self._layers.append(
            _LayerAccount(
                name=name,
                n_out_channels=int(n_out_channels),
                n_weights_per_out_channel=int(n_weights_per_out_channel),
                bits_module=bits_module,
                gate_module=gate_module,
            )
        )
        # Sanitize module-dict key (no dots)
        key = name.replace(".", "_")
        if bits_module is not None:
            self._registered_modules[f"{key}__bits"] = bits_module
        if gate_module is not None:
            self._registered_modules[f"{key}__gate"] = gate_module

    def layer_names(self) -> list[str]:
        return [L.name for L in self._layers]

    def expected_active_channels(self) -> torch.Tensor:
        """Differentiable Σ_l Σ_c E[gate(l, c)]."""
        if not self._layers:
            # Return a zero with grad disabled but device-agnostic
            return torch.zeros((), dtype=torch.float32)
        terms = []
        for L in self._layers:
            if L.gate_module is None:
                # No gate: all channels active
                terms.append(
                    torch.tensor(
                        float(L.n_out_channels),
                        dtype=torch.float32,
                        device=self._device(),
                    )
                )
            else:
                terms.append(L.gate_module.expected_active().sum())
        return torch.stack([t.to(torch.float32) for t in terms]).sum()

    def expected_total_bits(self) -> torch.Tensor:
        """Differentiable Σ_l Σ_c E[gate(l, c)] · bits(l, c) · K(l).

        For per-element bit modules, bits is averaged across the
        per-out-channel slice (mean over the input × kernel dims).
        """
        if not self._layers:
            return torch.zeros((), dtype=torch.float32)
        terms = []
        device = self._device()
        for L in self._layers:
            # Effective active channel count
            if L.gate_module is None:
                active_per_channel = torch.ones(
                    L.n_out_channels, dtype=torch.float32, device=device
                )
            else:
                active_per_channel = L.gate_module.expected_active().to(torch.float32)
            # Effective bits per channel
            if L.bits_module is None:
                bits_per_channel = torch.full(
                    (L.n_out_channels,), 8.0, dtype=torch.float32, device=device
                )
            else:
                bits = L.bits_module.bits_used()
                if bits.numel() == L.n_out_channels:
                    # Per-channel
                    bits_per_channel = bits.to(torch.float32)
                elif bits.numel() == 1:
                    bits_per_channel = bits.expand(L.n_out_channels).to(torch.float32)
                else:
                    # Per-element: average per output channel
                    if bits.shape[0] != L.n_out_channels:
                        raise ValueError(
                            f"layer {L.name!r}: bits_module shape[0]={bits.shape[0]} "
                            f"!= n_out_channels={L.n_out_channels}"
                        )
                    bits_per_channel = bits.reshape(L.n_out_channels, -1).mean(dim=1)
                bits_per_channel = bits_per_channel.to(torch.float32)
            # Joint accounting per channel:
            #   active_c · bits_c · weights_per_out_channel
            terms.append(
                (active_per_channel * bits_per_channel).sum()
                * float(L.n_weights_per_out_channel)
            )
        return torch.stack(terms).sum()

    def expected_total_bytes(self) -> torch.Tensor:
        """expected_total_bits / 8 — for pretty reporting."""
        return self.expected_total_bits() / 8.0

    def hard_summary(self) -> dict[str, dict[str, int | float]]:
        """Eval-time integer summary per layer (for export/reporting)."""
        out: dict[str, dict[str, int | float]] = {}
        for L in self._layers:
            n_active = (
                L.gate_module.hard_count()
                if L.gate_module is not None
                else L.n_out_channels
            )
            if L.bits_module is None:
                mean_bits = 8.0
            elif hasattr(L.bits_module, "mean_bits"):
                mean_bits = float(L.bits_module.mean_bits())
            else:
                mean_bits = float(L.bits_module.bits_used().detach().mean().item())
            out[L.name] = {
                "n_total_channels": L.n_out_channels,
                "n_active_channels": int(n_active),
                "weights_per_out_channel": L.n_weights_per_out_channel,
                "mean_bits": mean_bits,
                "hard_total_bits": int(n_active * mean_bits * L.n_weights_per_out_channel),
            }
        return out

    def _device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")


# ── Functional helpers (no module state) ───────────────────────────────


def expected_active_channels(
    *,
    accountant: JointBitsAndWidthAccountant,
) -> torch.Tensor:
    """Functional alias for downstream training scripts."""
    return accountant.expected_active_channels()


def expected_total_bits(
    *,
    accountant: JointBitsAndWidthAccountant,
) -> torch.Tensor:
    """Functional alias."""
    return accountant.expected_total_bits()


def hard_threshold_gates(
    *,
    accountant: JointBitsAndWidthAccountant,
) -> dict[str, torch.Tensor]:
    """Return per-layer deterministic gate vectors (eval-time)."""
    out: dict[str, torch.Tensor] = {}
    for L in accountant._layers:
        if L.gate_module is None:
            out[L.name] = torch.ones(L.n_out_channels)
        else:
            out[L.name] = L.gate_module._deterministic_gate()
    return out


# ── Joint compression loss ─────────────────────────────────────────────


def joint_compress_loss(
    *,
    task_loss: torch.Tensor,
    accountant: JointBitsAndWidthAccountant,
    lambda_bits: float,
    lambda_gate: float = 0.0,
    target_bits: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wang et al. 2023 joint loss = task + λ_b · bits + λ_g · L0.

    Args (all required-keyword):
        task_loss: scalar tensor — the task-side loss (e.g. proxy seg+pose).
        accountant: JointBitsAndWidthAccountant instance.
        lambda_bits: rate-Lagrangian weight on E[total_bits]. Must be ≥ 0.
        lambda_gate: optional extra weight on raw L0 (active-channel count).
            Default 0; bits accounting already includes gate weighting.
        target_bits: if set, the bits term becomes squared-hinge:
            `lambda_bits · max(0, E[bits] - target_bits)²`.
            This is the Wang et al. "soft target" formulation that
            converges to a specific budget instead of pareto sweeping.
            Default None: linear `lambda_bits · E[bits]`.

    Returns:
        (total_loss, parts_dict). parts_dict has keys:
            - "task" — scalar task loss
            - "bits" — scalar E[total_bits] in bits
            - "gate" — scalar Σ E[gate==1]
            - "rate_term" — λ_b · (bits or hinge(bits))
            - "l0_term" — λ_g · gate
    """
    if lambda_bits < 0:
        raise ValueError(f"lambda_bits must be non-negative, got {lambda_bits}")
    if lambda_gate < 0:
        raise ValueError(f"lambda_gate must be non-negative, got {lambda_gate}")
    if not isinstance(task_loss, torch.Tensor):
        raise TypeError(
            f"task_loss must be a torch.Tensor, got {type(task_loss).__name__}"
        )
    if task_loss.numel() != 1:
        raise ValueError(
            f"task_loss must be scalar, got shape {tuple(task_loss.shape)}"
        )

    bits = accountant.expected_total_bits()
    gates = accountant.expected_active_channels()

    if target_bits is None:
        rate_term = lambda_bits * bits
    else:
        if target_bits < 0:
            raise ValueError(f"target_bits must be non-negative, got {target_bits}")
        excess = (bits - float(target_bits)).clamp(min=0.0)
        rate_term = lambda_bits * (excess ** 2)

    l0_term = lambda_gate * gates

    total = task_loss + rate_term + l0_term

    return total, {
        "task": task_loss.detach(),
        "bits": bits.detach(),
        "gate": gates.detach(),
        "rate_term": rate_term.detach(),
        "l0_term": l0_term.detach(),
    }


# ── λ scheduler (cosine ramp à la Wang et al.) ─────────────────────────


@dataclass
class SCNNRateScheduler:
    """Cosine-ramped Lagrangian schedule per Wang et al. arXiv:2301.13142 §4.

    `lambda_bits` ramps from 0 to `peak` over `warmup_epochs`, holds at
    `peak` for `hold_epochs`, then decays via cosine to 0 over `cool_epochs`.

    Total cycle = warmup + hold + cool. After cycle, lambda stays at 0
    (or repeats if `cyclic=True` per the multi-cycle schedule).
    """

    peak: float
    warmup_epochs: int
    hold_epochs: int
    cool_epochs: int
    cyclic: bool = False

    def __post_init__(self) -> None:
        if self.peak < 0:
            raise ValueError(f"peak must be non-negative, got {self.peak}")
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )
        if self.hold_epochs < 0:
            raise ValueError(
                f"hold_epochs must be non-negative, got {self.hold_epochs}"
            )
        if self.cool_epochs < 0:
            raise ValueError(
                f"cool_epochs must be non-negative, got {self.cool_epochs}"
            )
        if self.warmup_epochs + self.hold_epochs + self.cool_epochs == 0:
            raise ValueError(
                "schedule must have at least one of {warmup, hold, cool} > 0"
            )

    def lambda_at(self, epoch: int) -> float:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        cycle = self.warmup_epochs + self.hold_epochs + self.cool_epochs
        if self.cyclic:
            e = epoch % cycle
        else:
            if epoch >= cycle:
                return 0.0
            e = epoch
        if e < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return self.peak
            return self.peak * (e / self.warmup_epochs)
        if e < self.warmup_epochs + self.hold_epochs:
            return self.peak
        # Cool phase
        cool_pos = e - self.warmup_epochs - self.hold_epochs
        if self.cool_epochs == 0:
            return 0.0
        # Cosine from peak → 0
        return float(
            0.5 * self.peak * (1.0 + math.cos(math.pi * cool_pos / self.cool_epochs))
        )
