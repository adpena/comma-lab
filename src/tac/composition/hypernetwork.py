# SPDX-License-Identifier: MIT
"""Hypernetwork Composer — composition/stacking primitive.

Ha-Dai-Le 2016 ("HyperNetworks", https://arxiv.org/abs/1609.09106) showed
that a small "hyper" network ``h(z; θ_h)`` can GENERATE the weights of a
larger "target" network ``f(x; θ_f)``. If the hyper is small enough, the
TOTAL parameter count ``|θ_h| + |z|·N`` (with N latent codes) is strictly
less than the target's full ``|θ_f|``.

This module implements a deterministic MLP-style hypernetwork primitive
that maps a small latent code ``z ∈ R^d_z`` through a 2-layer MLP into a
flat parameter vector ``θ_f`` of any requested shape:

    θ_f = W_2 · GELU(W_1 · z + b_1) + b_2                                 (H.1)

For per-pair conditional weights (e.g. one latent per pair-of-frames in the
contest video), ``N`` codes are stored alongside the hyper-MLP. The total
on-disk cost is

    |θ_h| + N · d_z

For N=600 pairs, d_z=4, hidden=8, target=K params: cost = 4·K + 600·4 + ~100
which beats raw storage iff K > 1, i.e. always.

Source memos:
- Hypernetwork concept: Ha-Dai-Le 2016 (cited in ancient-elder Era 10).
- Composer pattern: ``.omx/research/ancient_elder_polymath_research_20260513.md``.

Cross-references
----------------
- Distillation chain (``tac.composition.distillation_chain``) — chain
  hypernetworks for hierarchical compression.
- Trainer skeleton: ``tac.substrates._shared.trainer_skeleton``.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
This module produces a forward-pass module + serialisable spec; it does not
modify archive bytes by itself. Substrate integration must register a
parser-section manifest entry per CLAUDE.md Catalog #124 before any
``score_claim=True``. Until paired ``[contest-CUDA]`` + ``[contest-CPU]``
anchors land on a Hypernet-equipped substrate, every result is
``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False``.

HNeRV parity discipline (13 lessons)
------------------------------------
1. Score-aware: hypernet is fully differentiable; trainer drives
   apply_eval_roundtrip + scorer gradient.
2. Export-first: :meth:`Hypernetwork.serialize_state` is deterministic.
3-6. Substrate concerns; not violated.
7. Bolt-on ≤ 350 LOC.
8-13. Standard substrate concerns; not violated.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch
from torch import nn

HYPER_MAGIC = b"HYP1"
HYPER_SCHEMA_VERSION = 1
MAX_OUTPUT_DIM = 1 << 24  # 16M params per latent — safety bound.


class HypernetworkError(ValueError):
    """Raised when a Hypernetwork spec or input is invalid."""


@dataclass(frozen=True)
class HypernetworkSpec:
    """Specification for an MLP-style hypernetwork.

    Args:
        latent_dim: dimensionality ``d_z`` of the input latent code.
        hidden_dim: hidden-layer width of the 2-layer MLP. Must be > 0.
        output_dim: number of FLOAT parameters the hyper emits per latent
            (i.e. the size of the flat ``θ_f`` it generates).
        num_codes: number of latent codes ``N`` stored alongside the
            hyper. Use 1 for unconditional, 600 for per-contest-pair, etc.
        activation: non-linearity name. One of {"gelu", "relu", "tanh"}.
        init_scale: per-layer init scale (Kaiming-like).
    """

    latent_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 1024
    num_codes: int = 1
    activation: str = "gelu"
    init_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise HypernetworkError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.hidden_dim <= 0:
            raise HypernetworkError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.output_dim <= 0 or self.output_dim > MAX_OUTPUT_DIM:
            raise HypernetworkError(
                f"output_dim must be in (0, {MAX_OUTPUT_DIM}], got {self.output_dim}"
            )
        if self.num_codes <= 0:
            raise HypernetworkError(f"num_codes must be positive, got {self.num_codes}")
        if self.activation not in {"gelu", "relu", "tanh"}:
            raise HypernetworkError(
                f"activation must be in 'gelu'/'relu'/'tanh', got {self.activation!r}"
            )
        if self.init_scale <= 0 or not math.isfinite(float(self.init_scale)):
            raise HypernetworkError(f"init_scale must be positive, got {self.init_scale}")


def _make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise HypernetworkError(f"unknown activation: {name}")


class Hypernetwork(nn.Module):
    """Generate target parameters from a latent code via 2-layer MLP.

    Architecture:
        ``z`` (d_z) → Linear(d_z → hidden) → activation → Linear(hidden → output_dim)

    Latent codes are stored as an ``nn.Parameter`` of shape ``(N, d_z)``.

    Example
    -------
    >>> import torch
    >>> from tac.composition.hypernetwork import Hypernetwork, HypernetworkSpec
    >>> spec = HypernetworkSpec(latent_dim=2, hidden_dim=4, output_dim=8, num_codes=3)
    >>> hyper = Hypernetwork(spec)
    >>> theta = hyper(torch.tensor([0]))
    >>> theta.shape
    torch.Size([1, 8])
    """

    def __init__(self, spec: HypernetworkSpec) -> None:
        super().__init__()
        self.spec = spec
        self.fc1 = nn.Linear(spec.latent_dim, spec.hidden_dim)
        self.fc2 = nn.Linear(spec.hidden_dim, spec.output_dim)
        self.act = _make_activation(spec.activation)
        self.codes = nn.Parameter(
            torch.randn(spec.num_codes, spec.latent_dim) * 0.1
        )
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            fan_in_1 = self.spec.latent_dim
            fan_in_2 = self.spec.hidden_dim
            self.fc1.weight.mul_(self.spec.init_scale / max(1.0, math.sqrt(fan_in_1)))
            self.fc1.bias.zero_()
            self.fc2.weight.mul_(self.spec.init_scale / max(1.0, math.sqrt(fan_in_2)))
            self.fc2.bias.zero_()

    def forward(self, code_index: torch.Tensor) -> torch.Tensor:
        """Generate target params for a batch of latent-code indices.

        Args:
            code_index: 1-D LongTensor of shape ``(B,)`` containing indices
                into ``self.codes`` (in ``[0, num_codes)``).

        Returns:
            Tensor of shape ``(B, output_dim)`` of generated target params.
        """
        if code_index.dim() != 1:
            raise HypernetworkError(
                f"code_index must be 1-D, got dim {code_index.dim()}"
            )
        if not torch.is_tensor(code_index) or code_index.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise HypernetworkError(
                f"code_index must be int32/int64 LongTensor, got dtype {code_index.dtype}"
            )
        max_idx = int(code_index.max().item()) if code_index.numel() > 0 else -1
        min_idx = int(code_index.min().item()) if code_index.numel() > 0 else 0
        if min_idx < 0 or max_idx >= self.spec.num_codes:
            raise HypernetworkError(
                f"code_index out of range [0, {self.spec.num_codes}); "
                f"got [{min_idx}, {max_idx}]"
            )
        z = self.codes[code_index]  # (B, d_z)
        h = self.act(self.fc1(z))
        return self.fc2(h)

    def generate_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Generate target params directly from a continuous latent ``z``.

        Use this for unconditional generation or continuous interpolation
        between learned codes.

        Args:
            z: tensor of shape ``(..., d_z)``.

        Returns:
            Tensor of shape ``(..., output_dim)``.
        """
        if z.shape[-1] != self.spec.latent_dim:
            raise HypernetworkError(
                f"z.shape[-1]={z.shape[-1]} must equal latent_dim="
                f"{self.spec.latent_dim}"
            )
        if not torch.isfinite(z).all():
            raise HypernetworkError("z must contain finite values")
        h = self.act(self.fc1(z))
        return self.fc2(h)

    def estimate_total_param_bytes(self, dtype_bytes: int = 4) -> int:
        """Total params * dtype_bytes (MLP weights + codes)."""
        n = (
            self.spec.latent_dim * self.spec.hidden_dim
            + self.spec.hidden_dim  # fc1.bias
            + self.spec.hidden_dim * self.spec.output_dim
            + self.spec.output_dim  # fc2.bias
            + self.spec.num_codes * self.spec.latent_dim  # codes
        )
        return n * dtype_bytes

    def serialize_state(self) -> bytes:
        """Deterministic serialisation: spec header then state_dict tensors."""
        body = bytearray()
        body += HYPER_MAGIC
        body += struct.pack("<H", HYPER_SCHEMA_VERSION)
        body += struct.pack("<I", self.spec.latent_dim)
        body += struct.pack("<I", self.spec.hidden_dim)
        body += struct.pack("<I", self.spec.output_dim)
        body += struct.pack("<I", self.spec.num_codes)
        act_bytes = self.spec.activation.encode("ascii")
        body += struct.pack("<H", len(act_bytes))
        body += act_bytes
        body += struct.pack("<d", self.spec.init_scale)
        # Serialise tensors in deterministic order.
        for tensor in (
            self.fc1.weight.detach().cpu().contiguous(),
            self.fc1.bias.detach().cpu().contiguous(),
            self.fc2.weight.detach().cpu().contiguous(),
            self.fc2.bias.detach().cpu().contiguous(),
            self.codes.detach().cpu().contiguous(),
        ):
            arr = tensor.to(torch.float32).numpy().tobytes()
            body += struct.pack("<I", len(arr))
            body += arr
        return bytes(body)

    @classmethod
    def deserialize_state(cls, payload: bytes) -> Hypernetwork:
        """Inverse of :meth:`serialize_state`."""
        if len(payload) < 4 or payload[:4] != HYPER_MAGIC:
            raise HypernetworkError(f"bad magic: {payload[:4]!r}")
        off = 4
        (version,) = struct.unpack_from("<H", payload, off)
        off += 2
        if version != HYPER_SCHEMA_VERSION:
            raise HypernetworkError(f"unsupported schema version: {version}")
        (latent_dim,) = struct.unpack_from("<I", payload, off)
        off += 4
        (hidden_dim,) = struct.unpack_from("<I", payload, off)
        off += 4
        (output_dim,) = struct.unpack_from("<I", payload, off)
        off += 4
        (num_codes,) = struct.unpack_from("<I", payload, off)
        off += 4
        (act_len,) = struct.unpack_from("<H", payload, off)
        off += 2
        activation = payload[off : off + act_len].decode("ascii")
        off += act_len
        (init_scale,) = struct.unpack_from("<d", payload, off)
        off += 8
        spec = HypernetworkSpec(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_codes=num_codes,
            activation=activation,
            init_scale=init_scale,
        )
        hyper = cls(spec)
        shapes = [
            (hidden_dim, latent_dim),
            (hidden_dim,),
            (output_dim, hidden_dim),
            (output_dim,),
            (num_codes, latent_dim),
        ]
        tensors = []
        for shape in shapes:
            (nbytes,) = struct.unpack_from("<I", payload, off)
            off += 4
            n_floats = nbytes // 4
            expected = 1
            for s in shape:
                expected *= s
            if n_floats != expected:
                raise HypernetworkError(
                    f"shape {shape} mismatch: payload has {n_floats}, expected {expected}"
                )
            buf = payload[off : off + nbytes]
            off += nbytes
            t = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(shape).clone()
            tensors.append(t)
        with torch.no_grad():
            hyper.fc1.weight.copy_(tensors[0])
            hyper.fc1.bias.copy_(tensors[1])
            hyper.fc2.weight.copy_(tensors[2])
            hyper.fc2.bias.copy_(tensors[3])
            hyper.codes.copy_(tensors[4])
        return hyper


__all__ = [
    "HYPER_MAGIC",
    "HYPER_SCHEMA_VERSION",
    "Hypernetwork",
    "HypernetworkError",
    "HypernetworkSpec",
]
