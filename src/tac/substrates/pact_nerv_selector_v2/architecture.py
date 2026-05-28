# SPDX-License-Identifier: MIT
"""pact_nerv_selector_v2 architecture - Pact-NeRV-SELECTOR-V2 (L0 SKETCH).

HNeRV-class implicit renderer + arithmetic-coded selector index per pair.
The distinguishing primitive vs FEC6 fixed-Huffman k=16: ARITHMETIC coding
over the same 16-mode palette achieves fractional-bit precision (Witten 1987)
vs integer-bit Huffman code-lengths. Sister of fec6 selector at commit
``a8970e36`` (FEC6 fixed-Huffman k=16 frame-exploit selector).

L0 SCAFFOLD posture: the base HNeRV decoder mirrors pact_nerv_ia3 /
boost_nerv. The arithmetic selector coder is the bolt-on under test.

CLAUDE.md compliance:
- No silent device defaults
- No scorer load at inflate time
- No /tmp paths
- Reviewable in 30 seconds per HNeRV parity L12
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONTEST_H = 384
_CONTEST_W = 512
_NUM_FRAMES = 1200
_PAIRS = _NUM_FRAMES // 2

FEC6_FIXED_K16_MODE_IDS: tuple[str, ...] = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)


@dataclass(frozen=True)
class PactNervSelectorV2Config:
    """Static design-time parameters for Pact-NeRV-SELECTOR-V2."""

    latent_dim: int = 24
    embed_dim: int = 64
    initial_grid_h: int = 3
    initial_grid_w: int = 4
    decoder_channels: tuple[int, ...] = (48, 40, 32, 24, 20, 16, 12)
    sin_frequency: float = 30.0
    num_upsample_blocks: int = 7
    num_pairs: int = _PAIRS
    output_height: int = _CONTEST_H
    output_width: int = _CONTEST_W

    selector_palette_size: int = 16
    """FEC6 k=16 palette per CROSS-CANDIDATE finding #1 empirical anchor."""


class _SinAct(nn.Module):
    def __init__(self, w: float) -> None:
        super().__init__()
        self.w = float(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w * x)


class _DepthSepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class _DsUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        super().__init__()
        self.dsc = _DepthSepConv(in_ch, out_ch * 4)
        self.act = _SinAct(sin_freq)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.act(self.dsc(x)))


class ArithmeticSelectorCoder:
    """Static arithmetic coder for FEC6 k=16 selector indices (Witten 1987).

    Per the canonical integer arithmetic-coding algorithm (Witten-Neal-Cleary
    1987, modified per Said 2004 §6 for 32-bit precision): symbols drawn
    from a fixed-size alphabet with KNOWN cumulative frequencies are
    encoded into a sub-interval of [0, 2**precision). The encoded bitstream
    achieves average code-length within 2 bits of the source entropy
    H(X) = -sum p(x) log2 p(x) regardless of whether p(x) are powers of 2.

    For the FEC6 k=16 palette, the static cum_freq table is derived from
    empirical frequencies measured on the contest video (in the canonical
    fec6 anchor). The L0 SCAFFOLD uses uniform p(x) = 1/16 as the design-
    time placeholder; L1 dispatch will fit the actual table to the contest
    frame-exploit selector distribution.

    Static table = HARD-EARNED-LITERATURE per Witten 1987 §3.2;
    uniform-init at L0 = CARGO-CULTED (CL1 dispatch will measure).
    """

    def __init__(
        self,
        palette_size: int,
        cum_freq: tuple[int, ...] | None = None,
        precision: int = 32,
    ) -> None:
        if palette_size < 2:
            raise ValueError(f"palette_size must be >= 2; got {palette_size}")
        if precision < 16 or precision > 62:
            raise ValueError(f"precision must be in [16, 62]; got {precision}")
        self.palette_size = palette_size
        self.precision = precision
        if cum_freq is None:
            # Uniform default per L0 SCAFFOLD; L1 dispatch fits to contest
            cum_freq = tuple(i + 1 for i in range(palette_size))
        if len(cum_freq) != palette_size:
            raise ValueError(
                f"cum_freq must have palette_size entries; got {len(cum_freq)}"
            )
        if any(cum_freq[i] >= cum_freq[i + 1] for i in range(palette_size - 1)):
            raise ValueError("cum_freq must be strictly monotonically increasing")
        if cum_freq[0] < 1:
            raise ValueError("cum_freq[0] must be >= 1 (no zero-probability symbols)")
        self.cum_freq = cum_freq
        self.total_freq = cum_freq[-1]

    def encode(self, symbols: list[int]) -> bytes:
        """Encode a symbol stream into the arithmetic-coded bitstream.

        Returns deterministic bytes. The L0 SCAFFOLD reference uses a
        simplified renormalized integer arithmetic coder per Said 2004 §6;
        a production-grade implementation would use the canonical range
        coder primitive at tac.entropy.range_codec (L1 wire-in pending).
        """
        if not symbols:
            return b""
        low = 0
        high = (1 << self.precision) - 1
        out_bits: list[int] = []
        pending = 0
        for sym in symbols:
            if sym < 0 or sym >= self.palette_size:
                raise ValueError(
                    f"symbol {sym} out of palette [0, {self.palette_size})"
                )
            sym_low = self.cum_freq[sym - 1] if sym > 0 else 0
            sym_high = self.cum_freq[sym]
            rng = high - low + 1
            high = low + (rng * sym_high) // self.total_freq - 1
            low = low + (rng * sym_low) // self.total_freq
            while True:
                if high < (1 << (self.precision - 1)):
                    out_bits.append(0)
                    out_bits.extend([1] * pending)
                    pending = 0
                elif low >= (1 << (self.precision - 1)):
                    out_bits.append(1)
                    out_bits.extend([0] * pending)
                    pending = 0
                    low -= 1 << (self.precision - 1)
                    high -= 1 << (self.precision - 1)
                elif low >= (1 << (self.precision - 2)) and high < 3 * (
                    1 << (self.precision - 2)
                ):
                    pending += 1
                    low -= 1 << (self.precision - 2)
                    high -= 1 << (self.precision - 2)
                else:
                    break
                low <<= 1
                high = (high << 1) | 1
        pending += 1
        if low < (1 << (self.precision - 2)):
            out_bits.append(0)
            out_bits.extend([1] * pending)
        else:
            out_bits.append(1)
            out_bits.extend([0] * pending)
        # Pack bits to bytes
        while len(out_bits) % 8 != 0:
            out_bits.append(0)
        out = bytearray()
        for i in range(0, len(out_bits), 8):
            byte = 0
            for bit in out_bits[i:i + 8]:
                byte = (byte << 1) | (bit & 1)
            out.append(byte)
        return bytes(out)

    def decode(self, payload: bytes, *, symbol_count: int) -> list[int]:
        """Decode bytes produced by :meth:`encode`.

        The archive grammar stores the pair count separately, so the arithmetic
        payload does not spend header bytes on length. Padding bits after the
        final emitted bit are interpreted as zero, matching :meth:`encode`.
        """
        if symbol_count < 0:
            raise ValueError(f"symbol_count must be >= 0; got {symbol_count}")
        if symbol_count == 0:
            if payload:
                raise ValueError("non-empty arithmetic payload for zero symbols")
            return []
        if not payload:
            raise ValueError("empty arithmetic payload for non-empty stream")

        bit_pos = 0

        def _next_bit() -> int:
            nonlocal bit_pos
            if bit_pos >= len(payload) * 8:
                bit_pos += 1
                return 0
            byte = payload[bit_pos // 8]
            bit = (byte >> (7 - (bit_pos % 8))) & 1
            bit_pos += 1
            return int(bit)

        low = 0
        high = (1 << self.precision) - 1
        value = 0
        for _ in range(self.precision):
            value = (value << 1) | _next_bit()

        half = 1 << (self.precision - 1)
        quarter = 1 << (self.precision - 2)
        three_quarter = 3 * quarter
        out: list[int] = []
        for _ in range(symbol_count):
            rng = high - low + 1
            scaled = ((value - low + 1) * self.total_freq - 1) // rng
            sym = 0
            for candidate, sym_high in enumerate(self.cum_freq):
                sym_low = self.cum_freq[candidate - 1] if candidate > 0 else 0
                if sym_low <= scaled < sym_high:
                    sym = candidate
                    break
            else:  # pragma: no cover - defensive corruption guard.
                raise ValueError(f"arithmetic payload scaled value {scaled} out of range")

            sym_low = self.cum_freq[sym - 1] if sym > 0 else 0
            sym_high = self.cum_freq[sym]
            high = low + (rng * sym_high) // self.total_freq - 1
            low = low + (rng * sym_low) // self.total_freq
            out.append(sym)

            while True:
                if high < half:
                    pass
                elif low >= half:
                    low -= half
                    high -= half
                    value -= half
                elif low >= quarter and high < three_quarter:
                    low -= quarter
                    high -= quarter
                    value -= quarter
                else:
                    break
                low <<= 1
                high = (high << 1) | 1
                value = (value << 1) | _next_bit()
        return out

    def encoded_bit_length(self, symbols: list[int]) -> int:
        """Return the entropy estimate for a symbol stream.

        Per Witten 1987: arithmetic coding achieves within 2 bits of the
        ideal H(X). The estimate uses sum_i log2(total_freq / freq_i).
        """
        if not symbols:
            return 0
        bits = 0.0
        for sym in symbols:
            if sym < 0 or sym >= self.palette_size:
                raise ValueError(f"symbol {sym} out of palette")
            freq = (
                self.cum_freq[sym] - (self.cum_freq[sym - 1] if sym > 0 else 0)
            )
            bits += math.log2(self.total_freq / freq)
        return math.ceil(bits)


def _parse_signed_token(token: str) -> int:
    if token.startswith("p"):
        return int(token[1:])
    if token.startswith("m"):
        return -int(token[1:])
    return int(token)


def _mode_spec(mode_id: str) -> tuple[str, tuple[int, ...], int]:
    if mode_id == "none":
        return ("identity", (), 0)
    frame_index = 1 if mode_id.startswith("frame1_") else 0
    base = mode_id.replace("frame1_", "frame0_", 1)
    if base.startswith("frame0_luma_bias_"):
        value = int(base.removeprefix("frame0_luma_bias_"))
        return ("rgb_bias", (value, value, value), frame_index)
    if base.startswith("frame0_rgb_bias_"):
        params = tuple(
            _parse_signed_token(part)
            for part in base.removeprefix("frame0_rgb_bias_").split("_")
        )
        if len(params) != 3:
            raise ValueError(f"bad RGB selector mode {mode_id!r}")
        return ("rgb_bias", params, frame_index)
    if base.startswith("frame0_blue_chroma_amp_"):
        return (
            "blue_chroma",
            (int(base.removeprefix("frame0_blue_chroma_amp_")),),
            frame_index,
        )
    if base.startswith("frame0_roll_dx"):
        suffix = base.removeprefix("frame0_roll_dx")
        dx_token, dy_token = suffix.split("_dy", 1)
        return ("roll", (int(dx_token), int(dy_token)), frame_index)
    raise ValueError(f"unsupported selector mode {mode_id!r}")


def _blue_tile(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tile = torch.tensor(
        [
            [-1, 1, -1, 1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, 1, -1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [-1, 1, 1, -1, -1, 1, 1, -1],
        ],
        dtype=dtype,
        device=device,
    )
    reps_h = (height + 7) // 8
    reps_w = (width + 7) // 8
    return tile.repeat(reps_h, reps_w)[:height, :width]


def apply_selector_code_to_pair_frames_255(
    frames_2chw: torch.Tensor,
    selector_code: int,
    *,
    mode_ids: tuple[str, ...] = FEC6_FIXED_K16_MODE_IDS,
) -> torch.Tensor:
    """Apply one K=16 frame selector to a rendered pair in byte space.

    Input and output are float tensors shaped ``(2, 3, H, W)`` in RGB byte
    space. The transform matches the PR110/FEC6 selector semantics: apply the
    chosen deterministic mode after renderer decode, before PNG emission.
    """
    if frames_2chw.shape[0] != 2 or frames_2chw.shape[1] != 3:
        raise ValueError(f"frames_2chw must be (2, 3, H, W); got {tuple(frames_2chw.shape)}")
    if selector_code < 0 or selector_code >= len(mode_ids):
        raise ValueError(f"selector code {selector_code} out of range {len(mode_ids)}")
    family, params, frame_index = _mode_spec(mode_ids[selector_code])
    if family == "identity":
        return frames_2chw.clamp(0.0, 255.0).round()
    out = frames_2chw.clone()
    frame = out[frame_index]
    if family == "rgb_bias":
        delta = torch.tensor(params, dtype=out.dtype, device=out.device).view(3, 1, 1)
        out[frame_index] = frame + delta
    elif family == "blue_chroma":
        amp = float(params[0])
        _channels, height, width = frame.shape
        tile = _blue_tile(height, width, device=out.device, dtype=out.dtype)
        adjusted = frame.clone()
        adjusted[0].add_(tile * amp)
        adjusted[2].sub_(tile * amp)
        out[frame_index] = adjusted
    elif family == "roll":
        dx, dy = int(params[0]), int(params[1])
        out[frame_index] = torch.roll(frame, shifts=(dy, dx), dims=(1, 2))
    else:  # pragma: no cover - _mode_spec owns supported families.
        raise ValueError(f"unsupported selector family {family!r}")
    return out.clamp(0.0, 255.0).round()


class PactNervSelectorV2Substrate(nn.Module):
    """Pact-NeRV-SELECTOR-V2 renderer (L0 SKETCH).

    Input: pair index ``i in [0, num_pairs)``.
    Output: ``(rgb_0, rgb_1)`` both ``(B, 3, H, W)`` in [0, 1].

    The arithmetic-selector primitive itself does not enter the forward
    pass at L0 (it operates at archive-encode time on per-pair selector
    indices). The substrate's L1 path will read per-pair selectors from
    the archive and route the matching deterministic frame-0 transform
    on the rendered RGB.
    """

    def __init__(self, cfg: PactNervSelectorV2Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.latents = nn.Parameter(
            torch.empty(cfg.num_pairs, cfg.latent_dim).normal_(std=0.02)
        )
        self.latent_embed = nn.Linear(
            cfg.latent_dim,
            cfg.embed_dim * cfg.initial_grid_h * cfg.initial_grid_w,
        )

        channels = [cfg.embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= cfg.num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({cfg.num_upsample_blocks}) entries"
            )
        blocks: list[nn.Module] = []
        for i in range(cfg.num_upsample_blocks):
            blocks.append(_DsUpBlock(channels[i], channels[i + 1], cfg.sin_frequency))
        self.blocks = nn.ModuleList(blocks)

        final_ch = channels[cfg.num_upsample_blocks]
        self.head_rgb_0 = nn.Conv2d(final_ch, 3, kernel_size=1)
        self.head_rgb_1 = nn.Conv2d(final_ch, 3, kernel_size=1)

        # Per-pair selector buffer (filled at archive build time; L0 SCAFFOLD
        # default = zeros = "none" mode per FEC6 palette).
        self.register_buffer(
            "selectors", torch.zeros(cfg.num_pairs, dtype=torch.long)
        )

        self._siren_init()

    def _siren_init(self) -> None:
        w = self.cfg.sin_frequency
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    if m.groups > 1:
                        fan_in = m.kernel_size[0] * m.kernel_size[1]
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    bound = math.sqrt(6.0 / fan_in) / max(w, 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(
        self, pair_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(
                f"pair_indices out of range [0, {self.cfg.num_pairs})"
            )

        z = self.latents[pair_indices]
        h = self.latent_embed(z)
        h = h.view(
            -1,
            self.cfg.embed_dim,
            self.cfg.initial_grid_h,
            self.cfg.initial_grid_w,
        )
        for block in self.blocks:
            h = block(h)

        if h.shape[-2:] != (self.cfg.output_height, self.cfg.output_width):
            h = F.interpolate(
                h,
                size=(self.cfg.output_height, self.cfg.output_width),
                mode="bilinear",
                align_corners=False,
            )
        rgb_0 = torch.sigmoid(self.head_rgb_0(h))
        rgb_1 = torch.sigmoid(self.head_rgb_1(h))
        return rgb_0, rgb_1

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
