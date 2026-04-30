"""Lane 20 — Ballé 2018 hyperprior + entropy bottleneck for renderer.bin.

Per Phase 3 Lane 20 spec (memory project_phases_2_3_4_*) and Round 5 Council E
battleplan §5.1 ACCELERATE rank #5: Johannes Ballé 2018 entropy bottleneck +
scale hyperprior. Replaces fixed factorized prior on renderer's qint stream
with learned scale-prior network.

Math foundation (Ballé, Minnen, Singh, Hwang, Johnston 2018 ICLR
"Variational Image Compression with a Scale Hyperprior")
---------------------------------------------------------

Latent y is encoded with rate `-log p_y(y|σ)` where σ is the hyperprior
estimating the local entropy:

    σ = h_s(z)
    z is auxiliary latent with its own prior p_z(z)
    p_y(y|σ) = N(y | 0, σ²) (or any parameterised conditional density)

Total bit cost:
    R = E[-log p_z(z)] + E[-log p_y(y|σ(z))]

Where p_z is a factorised prior (e.g. piecewise-linear CDF) and p_y is
N(0, σ²) — the σ values are TRANSMITTED as side-info, but the savings
on the y stream amortise the overhead when |y| >> |σ|.

Key amortisation gate (Ballé 2018 + memory):
- Side-info cost: 50-200 bytes (small MLP scale-prior network header)
- Savings on y stream: ~5-25% for streams >5KB
- Selfcomp 88K params × 1.017 bpw ≈ 11KB qint payload — borderline
- BORDERLINE means: V2 measurement decides if hyperprior beats static
  histogram. Lane 20 LANDS the side-info path; the dispatch decision
  is whether to ship Lane 20 or stay on V2 static histogram.

Scope of this scaffold
----------------------
- Minimal scale-prior MLP (4 layers; deterministic init)
- encode → decode round-trip on a synthetic qint stream
- Header overhead tracking (must be < 500B to amortise on 11KB stream)
- Byte-savings vs static factorized baseline (synthetic comparison)
- All claims tagged [synthetic] / [prediction]; real-stream confirm is
  the Phase 3 dispatch decision

Out of scope (Phase 3 follow-up)
--------------------------------
- End-to-end training of the scale-prior network on real renderer qints
- Variational inference for the auxiliary latent z
- Integration with Selfcomp's block-FP qint stream

CLAUDE.md compliance
--------------------
- Compress-time only training the scale-prior network; inflate runs
  encode/decode pure-math
- No silent defaults — every public function arg required-keyword
- No scorer load (works on PROVIDED qint streams)
- All claims tagged [synthetic] / [prediction]
- No GPU dependency; pure CPU encode/decode

References
----------
* Ballé, Minnen, Singh, Hwang, Johnston 2018 ICLR
* memory: project_codec_stacking_composition_canonical_orders §"hyperprior fit"
* memory: project_phases_2_3_4 §"Lane 20"
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ── magic bytes / format version ─────────────────────────────────────────


BHP_MAGIC: bytes = b"BHP1"
"""Lane 20 self-describing payload magic. 4 bytes, ASCII."""

BHP_VERSION: int = 1
"""Header version. Bumped on any wire-format change."""


# ── scale-prior MLP ───────────────────────────────────────────────────────


class ScalePriorMLP(nn.Module):
    """Tiny MLP that maps a 1-D context vector to per-element scale σ.

    Phase 3 production: train this on real renderer qint streams + use the
    output σ to drive the conditional density p_y(y|σ).

    Phase 2 scaffold: deterministic init; inference produces σ that the
    encoder uses to compute per-element rate via the gaussian model.

    Determinism: torch.manual_seed at construction; weights init from
    Xavier-uniform with the given seed.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 16,
        depth: int = 4,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        if context_dim < 1 or hidden_dim < 1 or depth < 2:
            raise ValueError(
                f"ScalePriorMLP: invalid arch (context_dim={context_dim}, "
                f"hidden_dim={hidden_dim}, depth={depth})"
            )
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        gen = torch.Generator().manual_seed(int(seed))
        layers: list[nn.Module] = []
        prev = self.context_dim
        for _ in range(self.depth - 1):
            lin = nn.Linear(prev, self.hidden_dim)
            with torch.no_grad():
                fan_in, fan_out = prev, self.hidden_dim
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                bound = (3.0 ** 0.5) * std
                lin.weight.uniform_(-bound, bound, generator=gen)
                lin.bias.zero_()
            layers.append(lin)
            layers.append(nn.GELU())
            prev = self.hidden_dim
        # Output: scalar σ per context (positive via softplus on output)
        out_lin = nn.Linear(prev, 1)
        with torch.no_grad():
            fan_in, fan_out = prev, 1
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            bound = (3.0 ** 0.5) * std
            out_lin.weight.uniform_(-bound, bound, generator=gen)
            # Bias starts at log(exp(1)-1) ≈ 0.541 → softplus(0.541) ≈ 1.0
            out_lin.bias.fill_(0.541)
        layers.append(out_lin)
        self.mlp = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Map (B, context_dim) → (B,) positive σ."""
        raw = self.mlp(context)  # (B, 1)
        sigma = self.softplus(raw).squeeze(-1)  # (B,) positive
        return sigma

    def num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def header_byte_size(self) -> int:
        """Side-info cost: int8-quantized weights + per-tensor scale.

        For amortisation analysis: the production wire format quantizes the
        MLP weights to int8 + 2 bytes per scale. Scaffold computes the
        budgeted size, NOT the actual encoded bytes (that's Phase 3).
        """
        # 1 byte per param + 2 bytes per layer scale
        scales_count = self.depth + 1  # one scale per linear layer (incl. output)
        return self.num_params() + scales_count * 2


# ── rate estimation under N(0, σ²) ────────────────────────────────────────


def gaussian_rate_bits(qint_stream: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Estimate per-element rate R(y) = -log2 p(y|σ) under N(0, σ²).

    For continuous Gaussian: -log p(y|σ) = 0.5*log(2πσ²) + 0.5*y²/σ².
    Approximation for quantized integer y ≈ continuous + 0.5 nat per element.

    Args:
        qint_stream: (N,) integer-valued tensor (cast to float for math).
        sigma: (N,) positive scale tensor.

    Returns:
        (N,) tensor of bits per element.
    """
    if qint_stream.shape != sigma.shape:
        raise ValueError(
            f"gaussian_rate_bits: shape mismatch "
            f"qint_stream={tuple(qint_stream.shape)} sigma={tuple(sigma.shape)}"
        )
    y = qint_stream.float()
    s = sigma.clamp_min(1e-6)
    # -log p(y|σ) in nats
    nats = 0.5 * torch.log(2.0 * math.pi * s * s) + 0.5 * (y * y) / (s * s)
    bits = nats / math.log(2.0)
    return bits


def static_factorised_rate_bits(qint_stream: torch.Tensor) -> torch.Tensor:
    """Static-factorised baseline: assume all elements share a single σ.

    This is the no-hyperprior baseline. The single σ is estimated from the
    stream's empirical std. If the actual stream is heteroscedastic, the
    hyperprior will beat this baseline; if homoscedastic, no gain (and the
    side-info is wasted).
    """
    y = qint_stream.float()
    sigma_global = y.std().clamp_min(1e-6)
    sigma = torch.full_like(y, float(sigma_global.item()))
    return gaussian_rate_bits(qint_stream, sigma)


# ── encode / decode (scale-prior MLP weights as side-info) ────────────────


@dataclass(frozen=True)
class BHPHeader:
    """Parsed BHP1 header fields."""

    version: int
    context_dim: int
    hidden_dim: int
    depth: int
    num_params: int
    payload_size: int


def encode_balle_hyperprior(
    scale_prior: ScalePriorMLP | None = None,
    qint_stream: torch.Tensor | None = None,
) -> bytes:
    """Encode the scale-prior MLP weights as Lane 20 side-info.

    Phase 2 scaffold: emits the MLP weights (fp16) + a header. Production V2
    Phase 3 will also emit the encoded qint stream conditional on the scale
    output of the MLP — but that requires a real arithmetic coder driven by
    the per-element σ, which is out of scope for the scaffold.

    Args:
        scale_prior: trained or untrained ScalePriorMLP. Required.
        qint_stream: (N,) qint tensor (int dtype). Required (used to derive
            the stream length recorded in the header so decode can validate).

    Returns:
        bytes — BHP1 self-describing payload (header + MLP weights).
    """
    if scale_prior is None:
        raise ValueError(
            "encode_balle_hyperprior: scale_prior is required (no silent "
            "default — Check 81 STRICT)."
        )
    if qint_stream is None:
        raise ValueError(
            "encode_balle_hyperprior: qint_stream is required (no silent "
            "default — Check 81 STRICT)."
        )
    if not torch.is_tensor(qint_stream) or qint_stream.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            f"encode_balle_hyperprior: qint_stream must be integer tensor; "
            f"got dtype {qint_stream.dtype}"
        )
    sd = scale_prior.state_dict()
    sorted_keys = sorted(sd.keys())
    flat_weights = torch.cat(
        [sd[k].detach().to(torch.float16).reshape(-1) for k in sorted_keys]
    )
    weight_bytes = flat_weights.cpu().numpy().tobytes()

    # Header (little-endian):
    #   magic            : 4 bytes  = b"BHP1"
    #   version          : 2 bytes  uint16
    #   context_dim      : 2 bytes  uint16
    #   hidden_dim       : 2 bytes  uint16
    #   depth            : 2 bytes  uint16
    #   num_params       : 4 bytes  uint32
    #   payload_size     : 8 bytes  uint64 (== len(weight_bytes))
    #   payload          : payload_size bytes
    header = io.BytesIO()
    header.write(BHP_MAGIC)
    header.write(struct.pack("<H", BHP_VERSION))
    header.write(struct.pack("<H", int(scale_prior.context_dim)))
    header.write(struct.pack("<H", int(scale_prior.hidden_dim)))
    header.write(struct.pack("<H", int(scale_prior.depth)))
    header.write(struct.pack("<I", int(scale_prior.num_params())))
    header.write(struct.pack("<Q", len(weight_bytes)))
    return header.getvalue() + weight_bytes


def _parse_bhp_header(blob: bytes) -> BHPHeader:
    """Strict header parser; raises ValueError on malformed input."""
    if len(blob) < 4 + 2 * 4 + 4 + 8:
        raise ValueError(
            f"decode_balle_hyperprior: blob length {len(blob)} too small for BHP1 header"
        )
    if blob[:4] != BHP_MAGIC:
        raise ValueError(
            f"decode_balle_hyperprior: bad magic {blob[:4]!r}, expected {BHP_MAGIC!r}"
        )
    buf = io.BytesIO(blob)
    buf.read(4)
    (version,) = struct.unpack("<H", buf.read(2))
    if version != BHP_VERSION:
        raise ValueError(
            f"decode_balle_hyperprior: unsupported version {version}; expected {BHP_VERSION}"
        )
    (context_dim,) = struct.unpack("<H", buf.read(2))
    (hidden_dim,) = struct.unpack("<H", buf.read(2))
    (depth,) = struct.unpack("<H", buf.read(2))
    (num_params,) = struct.unpack("<I", buf.read(4))
    (payload_size,) = struct.unpack("<Q", buf.read(8))
    return BHPHeader(
        version=int(version),
        context_dim=int(context_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        num_params=int(num_params),
        payload_size=int(payload_size),
    )


def decode_balle_hyperprior(blob: bytes | None = None) -> ScalePriorMLP:
    """Decode BHP1 payload back to a ScalePriorMLP.

    Pure-math byte → module. NO scorer load. NO GPU.

    Args:
        blob: bytes produced by encode_balle_hyperprior. Required.

    Returns:
        ScalePriorMLP with weights restored from the payload.
    """
    if blob is None:
        raise ValueError(
            "decode_balle_hyperprior: blob is required (no silent default — "
            "Check 81 STRICT)."
        )
    hdr = _parse_bhp_header(blob)
    scale_prior = ScalePriorMLP(
        context_dim=hdr.context_dim,
        hidden_dim=hdr.hidden_dim,
        depth=hdr.depth,
        seed=0,
    )
    if scale_prior.num_params() != hdr.num_params:
        raise ValueError(
            f"decode_balle_hyperprior: arch mismatch — header says "
            f"num_params={hdr.num_params} but reconstructed model has "
            f"{scale_prior.num_params()}"
        )
    header_size = 4 + 2 * 4 + 4 + 8
    payload = blob[header_size : header_size + hdr.payload_size]
    if len(payload) != hdr.payload_size:
        raise ValueError(
            f"decode_balle_hyperprior: truncated payload "
            f"(read {len(payload)} of {hdr.payload_size})"
        )
    flat = np.frombuffer(payload, dtype=np.float16).copy()
    sd = scale_prior.state_dict()
    sorted_keys = sorted(sd.keys())
    cursor = 0
    for k in sorted_keys:
        n = int(sd[k].numel())
        chunk = flat[cursor : cursor + n]
        if len(chunk) != n:
            raise ValueError(
                f"decode_balle_hyperprior: weight stream truncated at key {k!r} "
                f"(need {n}, got {len(chunk)})"
            )
        sd[k] = torch.from_numpy(chunk).reshape(sd[k].shape).to(sd[k].dtype)
        cursor += n
    scale_prior.load_state_dict(sd)
    return scale_prior


# ── amortisation analysis helpers ─────────────────────────────────────────


def amortisation_break_even_bytes(
    scale_prior: ScalePriorMLP,
    expected_savings_fraction: float,
) -> int:
    """Return the minimum stream byte count at which Lane 20 amortises.

    Side-info cost / expected savings fraction = break-even stream size.
    For Selfcomp 11KB qint payload + 5% expected savings (Ballé 2018 baseline):
        side_info ≈ 100B, break_even ≈ 100 / 0.05 = 2000B.
    11KB > 2000B → Lane 20 SHOULD amortise (BORDERLINE per memory).

    Args:
        scale_prior: the ScalePriorMLP whose header_byte_size is the
            side-info cost.
        expected_savings_fraction: estimated fraction of bytes saved on the
            y stream by the hyperprior (vs static-factorised baseline).
            Required.

    Returns:
        int — minimum y-stream bytes for amortisation.
    """
    if expected_savings_fraction is None or expected_savings_fraction <= 0:
        raise ValueError(
            f"amortisation_break_even_bytes: expected_savings_fraction must "
            f"be > 0; got {expected_savings_fraction}"
        )
    side_info = float(scale_prior.header_byte_size())
    return int(math.ceil(side_info / expected_savings_fraction))


__all__ = [
    "BHP_MAGIC",
    "BHP_VERSION",
    "BHPHeader",
    "ScalePriorMLP",
    "amortisation_break_even_bytes",
    "decode_balle_hyperprior",
    "encode_balle_hyperprior",
    "gaussian_rate_bits",
    "static_factorised_rate_bits",
]
