# SPDX-License-Identifier: MIT
"""Experimental entropy-coded archive helpers.

This module explores arithmetic-coded archive components and learned-probability
model hooks. It is not a promoted archive format and it does not establish a
score or byte-reduction claim by itself. A candidate is meaningful only after
the learned model bytes, decoder bytes, archive member bytes, and deterministic
round-trip evidence beat the current packed baseline for the same stream.

Components:
1. video.mkv -- H.265/AV1 encoded (unchanged, already compressed)
2. postfilter weights -- self-compressed + arithmetic coded
3. masks -- already entropy coded (239 bytes, leave alone)
4. pose targets -- arithmetic coded with learned probability model
5. entropy model weights -- ~1-2KB shared decoder

Status: research/prototype. Treat reduction estimates as hypotheses until a
byte-closed archive and exact CUDA auth eval artifact exist.

Usage::

    from tac.entropy_archive import (
        NeuralEntropyModel,
        ArithmeticCoder,
        ArithmeticDecoder,
        build_entropy_archive,
        inflate_entropy_archive,
    )
"""

from __future__ import annotations

import bisect
import io
import json
import struct
import zipfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ArithmeticCoder",
    "ArithmeticDecoder",
    "NeuralEntropyModel",
    "build_entropy_archive",
    "compress_byte_stream",
    "decompress_byte_stream",
    "inflate_entropy_archive",
]


# ── Arithmetic coding (deterministic, integer-based) ────────────────────

# Uses 32-bit precision integer arithmetic for exact reproducibility.
# No floating-point rounding issues between encode and decode.

_PRECISION = 32
_WHOLE = 1 << _PRECISION
_HALF = _WHOLE >> 1
_QUARTER = _WHOLE >> 2


class ArithmeticCoder:
    """Integer arithmetic encoder.

    Encodes a sequence of symbols given their cumulative probability
    distribution (CDF). The CDF is specified as integer counts (frequencies)
    that sum to a fixed total.

    Deterministic: given the same symbols and CDFs, always produces
    the same byte stream.

    Args:
        freq_total: sum of all frequency counts (precision of CDF).
    """

    def __init__(self, freq_total: int = 1 << 16):
        self.freq_total = freq_total
        self.low = 0
        self.high = _WHOLE - 1
        self.pending = 0
        self.output_bits: list[int] = []

    def encode_symbol(self, cum_freq: int, freq: int) -> None:
        """Encode one symbol given its position in the CDF.

        Args:
            cum_freq: cumulative frequency of symbols before this one.
            freq: frequency count of this symbol.
        """
        assert freq > 0, "Cannot encode zero-probability symbol"
        assert cum_freq + freq <= self.freq_total

        rng = self.high - self.low + 1
        self.high = self.low + (rng * (cum_freq + freq)) // self.freq_total - 1
        self.low = self.low + (rng * cum_freq) // self.freq_total

        while True:
            if self.high < _HALF:
                self._output_bit(0)
            elif self.low >= _HALF:
                self._output_bit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < 3 * _QUARTER:
                self.pending += 1
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        """Flush pending bits and return the encoded byte stream."""
        self.pending += 1
        if self.low < _QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)

        # Pad to byte boundary
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)

        # Pack bits into bytes
        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(self.output_bits):
                    byte |= self.output_bits[i + j] << j
            result.append(byte)

        return bytes(result)

    def _output_bit(self, bit: int) -> None:
        self.output_bits.append(bit)
        while self.pending > 0:
            self.output_bits.append(1 - bit)
            self.pending -= 1


class ArithmeticDecoder:
    """Integer arithmetic decoder (exact inverse of ArithmeticCoder).

    Args:
        data: byte stream from ArithmeticCoder.finish().
        freq_total: must match the encoder's freq_total.
    """

    def __init__(self, data: bytes, freq_total: int = 1 << 16):
        self.freq_total = freq_total
        self.low = 0
        self.high = _WHOLE - 1

        # Convert bytes to bit stream
        self.bits: list[int] = []
        for byte in data:
            for j in range(8):
                self.bits.append((byte >> j) & 1)
        self.bit_pos = 0

        # Initialize value register
        self.value = 0
        for _ in range(_PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def decode_symbol(self, cum_freqs: list[int], freqs: list[int]) -> int:
        """Decode one symbol given the CDF table.

        Args:
            cum_freqs: cumulative frequency for each symbol (ascending).
            freqs: frequency count for each symbol.

        Returns:
            Index of decoded symbol.
        """
        rng = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * self.freq_total - 1) // rng

        # Clamp scaled to valid range [0, freq_total-1]
        scaled = max(0, min(scaled, self.freq_total - 1))

        # Find symbol via binary search on cum_freqs (O(log N) instead of O(N))
        # cum_freqs is sorted ascending; bisect_right gives the first index
        # whose cum_freq is > scaled, so sym = that index - 1.
        sym = bisect.bisect_right(cum_freqs, scaled) - 1
        sym = max(0, min(sym, len(cum_freqs) - 1))

        # Update range
        cum = cum_freqs[sym]
        freq = freqs[sym]
        self.high = self.low + (rng * (cum + freq)) // self.freq_total - 1
        self.low = self.low + (rng * cum) // self.freq_total

        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.value -= _HALF
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < 3 * _QUARTER:
                self.value -= _QUARTER
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._read_bit()

        return sym

    def _read_bit(self) -> int:
        if self.bit_pos < len(self.bits):
            bit = self.bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0  # Implicit zero padding


# ── Neural entropy model ────────────────────────────────────────────────


class NeuralEntropyModel(nn.Module):
    """Tiny neural network that predicts probability distributions.

    Input: context (position index, previous values)
    Output: probability distribution over quantized value range

    This model is SHARED between encoder and decoder (stored once in archive).
    Size: ~1-2KB for a 2-layer MLP.

    Shannon's theorem: if the model's predicted probabilities match
    the true distribution, arithmetic coding achieves the entropy bound.

    Args:
        context_size: number of context features.
        hidden: hidden layer width.
        num_symbols: number of quantized symbols.
    """

    def __init__(
        self,
        context_size: int = 4,
        hidden: int = 16,
        num_symbols: int = 256,
    ):
        super().__init__()
        self.context_size = context_size
        self.hidden = hidden
        self.num_symbols = num_symbols

        self.net = nn.Sequential(
            nn.Linear(context_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_symbols),
        )
        # Initialize output to uniform distribution
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict probability distribution over symbols.

        Args:
            context: (B, context_size) float tensor.

        Returns:
            (B, num_symbols) log-probabilities (log-softmax).
        """
        return F.log_softmax(self.net(context), dim=-1)

    def predict_freqs(self, context: torch.Tensor, freq_total: int = 1 << 16) -> tuple[list[int], list[int]]:
        """Predict integer frequency table for arithmetic coding.

        Args:
            context: (1, context_size) float tensor.
            freq_total: precision of CDF.

        Returns:
            (cum_freqs, freqs) lists for ArithmeticDecoder.
        """
        with torch.no_grad():
            logprobs = self.forward(context)  # (1, num_symbols)
            probs = logprobs.exp().squeeze(0)  # (num_symbols,)

            # Convert to integer frequencies
            # Ensure each symbol gets at least 1 count (avoid zero-probability)
            raw_freqs = (probs * (freq_total - self.num_symbols)).long() + 1
            # Adjust to exactly sum to freq_total: distribute residual
            # across the most probable symbols without breaking min=1
            diff = freq_total - raw_freqs.sum().item()
            if diff != 0:
                # Sort by probability descending; distribute residual
                sorted_indices = probs.argsort(descending=True)
                remaining = diff
                for idx in sorted_indices:
                    idx_int = idx.item()
                    if remaining == 0:
                        break
                    if remaining > 0:
                        raw_freqs[idx_int] += 1
                        remaining -= 1
                    else:  # remaining < 0
                        if raw_freqs[idx_int] > 1:
                            raw_freqs[idx_int] -= 1
                            remaining += 1

            # Safety: if extreme distributions leave a residual after the loop
            # (e.g. all symbols at min freq=1 and not enough to subtract),
            # add the shortfall to the most-probable symbol.
            final_diff = freq_total - raw_freqs.sum().item()
            if final_diff != 0:
                most_probable = probs.argmax().item()
                raw_freqs[most_probable] += final_diff
                # Ensure we didn't go below 1
                if raw_freqs[most_probable] < 1:
                    raw_freqs[most_probable] = 1

            freqs = raw_freqs.tolist()
            cum_freqs = [0]
            for f in freqs[:-1]:
                cum_freqs.append(cum_freqs[-1] + f)

            return cum_freqs, freqs

    def train_on_data(
        self,
        values: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> float:
        """Train the entropy model on observed data.

        Args:
            values: (N,) long tensor of symbol indices.
            epochs: training epochs.
            lr: learning rate.

        Returns:
            Final cross-entropy loss (nats).
        """
        N = values.shape[0]
        # Build context: position index + previous values
        contexts = torch.zeros(N, self.context_size, dtype=torch.float32)
        for i in range(N):
            contexts[i, 0] = i / max(N - 1, 1)  # normalized position
            for j in range(1, min(self.context_size, i + 1)):
                contexts[i, j] = values[i - j].float() / self.num_symbols

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        final_loss = 0.0

        for epoch in range(epochs):
            logprobs = self.forward(contexts)  # (N, num_symbols)
            loss = F.nll_loss(logprobs, values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch == epochs - 1:
                final_loss = loss.item()

        return final_loss

    def serialize(self) -> bytes:
        """Serialize the entropy model to compact bytes.

        Uses a safe manual format: JSON header + raw float32 tensor data.
        This avoids torch.save/torch.load pickle-based serialization which
        is unsafe on untrusted archive data (arbitrary code execution).
        """
        header = {
            "context_size": self.context_size,
            "hidden": self.hidden,
            "num_symbols": self.num_symbols,
            "keys": [],
        }
        tensor_blobs: list[bytes] = []
        for key, tensor in sorted(self.state_dict().items()):
            t = tensor.cpu().float().contiguous()
            header["keys"].append({
                "name": key,
                "shape": list(t.shape),
                "numel": t.numel(),
            })
            tensor_blobs.append(t.numpy().tobytes())

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        buf = io.BytesIO()
        buf.write(struct.pack("<I", len(header_json)))
        buf.write(header_json)
        for blob in tensor_blobs:
            buf.write(blob)
        return buf.getvalue()

    @classmethod
    def deserialize(cls, data: bytes) -> NeuralEntropyModel:
        """Deserialize an entropy model from bytes.

        Uses safe manual unpacking (no pickle/torch.load). Compatible with
        the serialize() format above.
        """
        import numpy as np

        buf = io.BytesIO(data)
        header_len = struct.unpack("<I", buf.read(4))[0]
        header = json.loads(buf.read(header_len).decode("utf-8"))

        model = cls(
            context_size=header["context_size"],
            hidden=header["hidden"],
            num_symbols=header["num_symbols"],
        )

        state_dict = {}
        for key_info in header["keys"]:
            numel = key_info["numel"]
            raw = buf.read(numel * 4)  # float32 = 4 bytes
            arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(key_info["shape"])
            state_dict[key_info["name"]] = torch.from_numpy(arr)

        model.load_state_dict(state_dict)
        return model


# ── Entropy-coded data encoding/decoding ────────────────────────────────


def encode_with_entropy_model(
    values: torch.Tensor,
    entropy_model: NeuralEntropyModel,
    freq_total: int = 1 << 16,
) -> bytes:
    """Encode a sequence of symbols using the neural entropy model.

    Args:
        values: (N,) long tensor of symbol indices in [0, num_symbols).
        entropy_model: trained NeuralEntropyModel.
        freq_total: arithmetic coding precision.

    Returns:
        Compressed byte stream.
    """
    entropy_model.eval()
    N = values.shape[0]
    coder = ArithmeticCoder(freq_total=freq_total)

    for i in range(N):
        # Build context
        ctx = torch.zeros(1, entropy_model.context_size)
        ctx[0, 0] = i / max(N - 1, 1)
        for j in range(1, min(entropy_model.context_size, i + 1)):
            ctx[0, j] = values[i - j].float() / entropy_model.num_symbols

        cum_freqs, freqs = entropy_model.predict_freqs(ctx, freq_total=freq_total)

        sym = values[i].item()
        coder.encode_symbol(cum_freqs[sym], freqs[sym])

    return coder.finish()


def decode_with_entropy_model(
    data: bytes,
    entropy_model: NeuralEntropyModel,
    count: int,
    freq_total: int = 1 << 16,
) -> torch.Tensor:
    """Decode a sequence of symbols using the neural entropy model.

    Args:
        data: compressed byte stream from encode_with_entropy_model.
        entropy_model: same NeuralEntropyModel used for encoding.
        count: number of symbols to decode.
        freq_total: must match encoder's freq_total.

    Returns:
        (count,) long tensor of decoded symbol indices.
    """
    entropy_model.eval()
    decoder = ArithmeticDecoder(data, freq_total=freq_total)
    values = []

    for i in range(count):
        ctx = torch.zeros(1, entropy_model.context_size)
        ctx[0, 0] = i / max(count - 1, 1)
        for j in range(1, min(entropy_model.context_size, i + 1)):
            # Use float() to match encoding context construction exactly
            ctx[0, j] = float(values[-j]) / entropy_model.num_symbols

        cum_freqs, freqs = entropy_model.predict_freqs(ctx, freq_total=freq_total)
        sym = decoder.decode_symbol(cum_freqs, freqs)
        values.append(sym)

    return torch.tensor(values, dtype=torch.long)


# ── Pose target encoding ────────────────────────────────────────────────


def quantize_pose_targets(
    pose_targets: torch.Tensor,
    num_symbols: int = 256,
) -> tuple[torch.Tensor, float, float]:
    """Quantize float pose targets to symbol indices.

    Args:
        pose_targets: (N, D) float tensor.
        num_symbols: quantization levels.

    Returns:
        (symbols, min_val, max_val) where symbols is (N*D,) long tensor.
    """
    flat = pose_targets.reshape(-1).float()
    min_val = flat.min().item()
    max_val = flat.max().item()
    rng = max(max_val - min_val, 1e-10)
    normalized = (flat - min_val) / rng  # [0, 1]
    symbols = (normalized * (num_symbols - 1)).round().long().clamp(0, num_symbols - 1)
    return symbols, min_val, max_val


def dequantize_pose_targets(
    symbols: torch.Tensor,
    shape: tuple[int, ...],
    min_val: float,
    max_val: float,
    num_symbols: int = 256,
) -> torch.Tensor:
    """Dequantize symbol indices back to float pose targets."""
    rng = max(max_val - min_val, 1e-10)
    normalized = symbols.float() / (num_symbols - 1)
    flat = normalized * rng + min_val
    return flat.reshape(shape)


# ── Weight encoding ─────────────────────────────────────────────────────


def quantize_weights(
    state_dict: dict[str, torch.Tensor],
    num_symbols: int = 256,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize a state dict to symbol indices for entropy coding.

    Per-tensor quantization to num_symbols levels. Returns flattened
    symbols and metadata for reconstruction.

    Args:
        state_dict: PyTorch state dict.
        num_symbols: quantization levels.

    Returns:
        (symbols, metadata) where symbols is (total_elements,) long
        and metadata contains shapes and scales.
    """
    all_symbols = []
    meta_entries = []

    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].float().cpu()
        flat = tensor.reshape(-1)
        abs_max = flat.abs().max().item()
        if abs_max < 1e-10:
            abs_max = 1e-10

        # Symmetric quantization: map [-abs_max, abs_max] -> [0, num_symbols-1]
        normalized = (flat / abs_max + 1.0) / 2.0  # [0, 1]
        symbols = (normalized * (num_symbols - 1)).round().long().clamp(0, num_symbols - 1)
        all_symbols.append(symbols)

        meta_entries.append({
            "key": key,
            "shape": list(tensor.shape),
            "count": int(flat.shape[0]),
            "abs_max": abs_max,
        })

    return torch.cat(all_symbols), {"entries": meta_entries, "num_symbols": num_symbols}


def dequantize_weights(
    symbols: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Dequantize symbols back to a state dict."""
    num_symbols = metadata["num_symbols"]
    state_dict = {}
    offset = 0

    for entry in metadata["entries"]:
        count = entry["count"]
        syms = symbols[offset:offset + count]
        offset += count

        abs_max = entry["abs_max"]
        normalized = syms.float() / (num_symbols - 1)
        flat = (normalized * 2.0 - 1.0) * abs_max
        state_dict[entry["key"]] = flat.reshape(entry["shape"])

    return state_dict


# ── Archive building ────────────────────────────────────────────────────


def build_entropy_archive(
    video_mkv: bytes | None = None,
    postfilter_weights: dict[str, torch.Tensor] | bytes | None = None,
    masks: torch.Tensor | bytes | None = None,
    pose_targets: torch.Tensor | None = None,
    entropy_model: NeuralEntropyModel | None = None,
    *,
    train_entropy_model: bool = True,
    entropy_epochs: int = 200,
    num_symbols: int = 256,
) -> bytes:
    """Build a maximally compressed archive.

    Components:
    1. video.mkv -- H.265/AV1 encoded (stored as-is)
    2. postfilter weights -- quantized + arithmetic coded
    3. masks -- stored as-is (already entropy coded at 239 bytes)
    4. pose targets -- quantized + arithmetic coded
    5. entropy model weights -- ~1-2KB shared decoder

    Args:
        video_mkv: raw video bytes (stored verbatim).
        postfilter_weights: state dict or pre-compressed bytes.
        masks: segmentation masks (tensor or pre-encoded bytes).
        pose_targets: (N, D) float tensor of PoseNet targets.
        entropy_model: pre-trained entropy model (optional).
        train_entropy_model: if True, train model on the data.
        entropy_epochs: epochs to train entropy model.
        num_symbols: quantization levels.

    Returns:
        Compressed archive bytes (zip format for compatibility).
    """
    if entropy_model is None:
        entropy_model = NeuralEntropyModel(
            context_size=4, hidden=16, num_symbols=num_symbols,
        )

    buf = io.BytesIO()
    # Use ZIP_STORED (no deflate) since we handle compression ourselves
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        # 1. Video MKV
        if video_mkv is not None:
            zf.writestr("video.mkv", video_mkv)

        # 2. Postfilter weights
        if postfilter_weights is not None:
            if isinstance(postfilter_weights, bytes):
                # Already compressed (e.g., from self_compress.py)
                zf.writestr("postfilter.bin", postfilter_weights)
            else:
                # Quantize and entropy-code the state dict
                symbols, weight_meta = quantize_weights(postfilter_weights, num_symbols)

                if train_entropy_model:
                    entropy_model.train_on_data(symbols, epochs=entropy_epochs)

                encoded = encode_with_entropy_model(symbols, entropy_model)
                meta_json = json.dumps(weight_meta, separators=(",", ":")).encode("utf-8")

                zf.writestr("weights_encoded.bin", encoded)
                zf.writestr("weights_meta.json", meta_json)

        # 3. Masks (store as-is)
        if masks is not None:
            if isinstance(masks, bytes):
                zf.writestr("masks.bin", masks)
            else:
                zf.writestr("masks.bin", masks.cpu().to(torch.uint8).numpy().tobytes())

        # 4. Pose targets
        if pose_targets is not None:
            symbols_pose, min_val, max_val = quantize_pose_targets(pose_targets, num_symbols)

            # Train a separate entropy model for pose
            pose_entropy = NeuralEntropyModel(
                context_size=4, hidden=16, num_symbols=num_symbols,
            )
            if train_entropy_model:
                pose_entropy.train_on_data(symbols_pose, epochs=entropy_epochs)

            encoded_pose = encode_with_entropy_model(symbols_pose, pose_entropy)

            pose_meta = {
                "shape": list(pose_targets.shape),
                "min_val": min_val,
                "max_val": max_val,
                "count": int(symbols_pose.shape[0]),
                "num_symbols": num_symbols,
            }
            zf.writestr("pose_encoded.bin", encoded_pose)
            zf.writestr("pose_meta.json", json.dumps(pose_meta, separators=(",", ":")).encode("utf-8"))
            zf.writestr("pose_entropy.bin", pose_entropy.serialize())

        # 5. Entropy model (shared for weights)
        if postfilter_weights is not None and not isinstance(postfilter_weights, bytes):
            zf.writestr("entropy_model.bin", entropy_model.serialize())

    return buf.getvalue()


def inflate_entropy_archive(archive: bytes) -> dict[str, Any]:
    """Decode the entropy-coded archive at inflate time.

    Returns:
        Dict with keys: 'video_mkv', 'postfilter_weights', 'masks',
        'pose_targets' (any may be None if not in archive).
    """
    result: dict[str, Any] = {
        "video_mkv": None,
        "postfilter_weights": None,
        "masks": None,
        "pose_targets": None,
    }

    with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
        names = zf.namelist()

        # 1. Video
        if "video.mkv" in names:
            result["video_mkv"] = zf.read("video.mkv")

        # 2. Postfilter weights
        if "postfilter.bin" in names:
            # Self-compressed format
            result["postfilter_weights"] = zf.read("postfilter.bin")
        elif "weights_encoded.bin" in names:
            # Entropy-coded state dict
            entropy_model = NeuralEntropyModel.deserialize(zf.read("entropy_model.bin"))
            weight_meta = json.loads(zf.read("weights_meta.json"))
            encoded = zf.read("weights_encoded.bin")

            total_count = sum(e["count"] for e in weight_meta["entries"])
            symbols = decode_with_entropy_model(
                encoded, entropy_model, total_count,
            )
            result["postfilter_weights"] = dequantize_weights(symbols, weight_meta)

        # 3. Masks
        if "masks.bin" in names:
            result["masks"] = zf.read("masks.bin")

        # 4. Pose targets
        if "pose_encoded.bin" in names:
            pose_entropy = NeuralEntropyModel.deserialize(zf.read("pose_entropy.bin"))
            pose_meta = json.loads(zf.read("pose_meta.json"))
            encoded_pose = zf.read("pose_encoded.bin")

            symbols_pose = decode_with_entropy_model(
                encoded_pose, pose_entropy, pose_meta["count"],
            )
            result["pose_targets"] = dequantize_pose_targets(
                symbols_pose,
                tuple(pose_meta["shape"]),
                pose_meta["min_val"],
                pose_meta["max_val"],
                pose_meta["num_symbols"],
            )

    return result


# ── Static byte-stream compression (no neural model) ──────────────────


def compress_byte_stream(data: bytes) -> bytes:
    """Compress raw bytes using arithmetic coding with a static frequency table.

    Builds a histogram of the input byte values and uses it as the frequency
    table for arithmetic coding. No neural model needed -- this is pure
    Shannon entropy compression for FP4/int8 weight blobs.

    Format of output:
        4 bytes: original length (uint32)
        256 * 4 bytes: frequency table (uint32 each, sum = freq_total)
        remaining: arithmetic-coded payload

    Args:
        data: raw bytes to compress (e.g., quantized weight blob).

    Returns:
        Compressed byte stream.
    """
    if len(data) == 0:
        return struct.pack("<I", 0)

    freq_total = 1 << 16

    # Build histogram of byte values
    histogram = [0] * 256
    for b in data:
        histogram[b] += 1

    # Convert histogram to integer frequencies summing to freq_total.
    # Ensure every symbol gets at least 1 count (arithmetic coding requires it).
    freqs = [0] * 256
    for i in range(256):
        if histogram[i] > 0:
            freqs[i] = max(1, int(histogram[i] / len(data) * (freq_total - 256)) + 1)
        else:
            freqs[i] = 1

    # Adjust to sum exactly to freq_total
    diff = freq_total - sum(freqs)
    if diff != 0:
        # Find the most frequent symbol and adjust
        most_freq = max(range(256), key=lambda i: histogram[i])
        freqs[most_freq] += diff
        if freqs[most_freq] < 1:
            freqs[most_freq] = 1

    # Build CDF
    cum_freqs = [0] * 256
    for i in range(1, 256):
        cum_freqs[i] = cum_freqs[i - 1] + freqs[i - 1]

    # Encode
    coder = ArithmeticCoder(freq_total=freq_total)
    for b in data:
        coder.encode_symbol(cum_freqs[b], freqs[b])
    payload = coder.finish()

    # Pack: original_length + freq_table + payload
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(data)))
    for f in freqs:
        buf.write(struct.pack("<I", f))
    buf.write(payload)
    return buf.getvalue()


def decompress_byte_stream(data: bytes) -> bytes:
    """Decompress bytes produced by compress_byte_stream.

    Args:
        data: compressed byte stream from compress_byte_stream.

    Returns:
        Original uncompressed bytes.
    """
    buf = io.BytesIO(data)
    original_len = struct.unpack("<I", buf.read(4))[0]
    if original_len == 0:
        return b""

    freq_total = 1 << 16

    # Read frequency table
    freqs = []
    for _ in range(256):
        freqs.append(struct.unpack("<I", buf.read(4))[0])

    # Build CDF
    cum_freqs = [0] * 256
    for i in range(1, 256):
        cum_freqs[i] = cum_freqs[i - 1] + freqs[i - 1]

    # Decode
    payload = buf.read()
    decoder = ArithmeticDecoder(payload, freq_total=freq_total)
    result = bytearray()
    for _ in range(original_len):
        sym = decoder.decode_symbol(cum_freqs, freqs)
        result.append(sym)

    return bytes(result)


# ── Smoke tests ─────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Run basic encode/decode round-trip checks."""
    print("entropy_archive: running smoke tests...")

    # 1. Arithmetic coder round-trip
    freq_total = 1 << 16
    # Simple uniform CDF: 256 symbols
    n_sym = 256
    freq_each = freq_total // n_sym
    cum_freqs = [i * freq_each for i in range(n_sym)]
    freqs = [freq_each] * n_sym
    # Fix last to absorb rounding
    freqs[-1] = freq_total - cum_freqs[-1]

    symbols = [42, 100, 0, 255, 128, 77]
    coder = ArithmeticCoder(freq_total=freq_total)
    for s in symbols:
        coder.encode_symbol(cum_freqs[s], freqs[s])
    encoded = coder.finish()

    decoder = ArithmeticDecoder(encoded, freq_total=freq_total)
    decoded = []
    for _ in symbols:
        decoded.append(decoder.decode_symbol(cum_freqs, freqs))
    assert decoded == symbols, f"Arithmetic coding mismatch: {decoded} != {symbols}"
    print(f"  arithmetic coding: OK ({len(encoded)} bytes for {len(symbols)} symbols)")

    # 2. Neural entropy model
    model = NeuralEntropyModel(context_size=4, hidden=16, num_symbols=256)
    test_values = torch.randint(0, 256, (50,))
    loss = model.train_on_data(test_values, epochs=50)
    assert loss < 10.0, f"Entropy model loss too high: {loss}"
    print(f"  entropy model training: OK (final loss={loss:.3f})")

    # 3. Entropy model serialization round-trip
    blob = model.serialize()
    restored = NeuralEntropyModel.deserialize(blob)
    test_ctx = torch.randn(1, 4)
    orig_out = model(test_ctx)
    rest_out = restored(test_ctx)
    diff = (orig_out - rest_out).abs().max().item()
    assert diff < 0.1, f"Entropy model round-trip error: {diff}"
    print(f"  entropy model serialize: OK ({len(blob)} bytes, diff={diff:.4f})")

    # 4. Encode/decode with entropy model
    test_values_short = torch.randint(0, 256, (20,))
    model2 = NeuralEntropyModel(context_size=4, hidden=16, num_symbols=256)
    model2.train_on_data(test_values_short, epochs=100)
    encoded2 = encode_with_entropy_model(test_values_short, model2)
    decoded2 = decode_with_entropy_model(encoded2, model2, len(test_values_short))
    assert torch.equal(test_values_short, decoded2), (
        f"Entropy encode/decode mismatch:\n  orig={test_values_short.tolist()}\n  dec={decoded2.tolist()}"
    )
    print(f"  neural entropy encode/decode: OK ({len(encoded2)} bytes for {len(test_values_short)} symbols)")

    # 5. Pose target quantize/dequantize
    pose = torch.randn(10, 6) * 0.1  # typical pose range
    symbols_pose, mn, mx = quantize_pose_targets(pose, 256)
    restored_pose = dequantize_pose_targets(symbols_pose, pose.shape, mn, mx, 256)
    pose_err = (pose - restored_pose).abs().max().item()
    assert pose_err < 0.01, f"Pose quantization error too large: {pose_err}"
    print(f"  pose quantization: OK (max error={pose_err:.6f})")

    # 6. Weight quantize/dequantize
    state = {"w1": torch.randn(8, 3, 3, 3), "b1": torch.randn(8)}
    syms, meta = quantize_weights(state, 256)
    restored_state = dequantize_weights(syms, meta)
    for k in state:
        err = (state[k] - restored_state[k]).abs().max().item()
        assert err < 0.1, f"Weight quant error for {k}: {err}"
    print("  weight quantization: OK")

    # 7. Full archive round-trip
    video = b"fake_video_data_12345"
    pose = torch.randn(10, 6) * 0.1
    weights = {"conv.weight": torch.randn(4, 3, 3, 3) * 0.1, "conv.bias": torch.randn(4) * 0.01}

    archive = build_entropy_archive(
        video_mkv=video,
        postfilter_weights=weights,
        pose_targets=pose,
        entropy_epochs=50,
        num_symbols=256,
    )
    print(f"  archive built: {len(archive)} bytes")

    inflated = inflate_entropy_archive(archive)
    assert inflated["video_mkv"] == video
    assert inflated["postfilter_weights"] is not None
    assert inflated["pose_targets"] is not None
    pose_err = (pose - inflated["pose_targets"]).abs().max().item()
    # Pose values go through quantize -> encode -> decode -> dequantize
    # 256 levels over the range gives ~1/256 quantization step
    rng = pose.max().item() - pose.min().item()
    expected_max_err = rng / 255 + 1e-6
    assert pose_err < expected_max_err + 0.01, f"Archive pose error: {pose_err} > expected {expected_max_err}"
    print(f"  archive round-trip: OK (pose error={pose_err:.6f}, expected max={expected_max_err:.6f})")

    # 8. Archive with pre-compressed weights (self_compress integration)
    archive2 = build_entropy_archive(
        video_mkv=video,
        postfilter_weights=b"precompressed_blob_data",
        pose_targets=pose,
        entropy_epochs=50,
    )
    inflated2 = inflate_entropy_archive(archive2)
    assert inflated2["postfilter_weights"] == b"precompressed_blob_data"
    print(f"  archive with pre-compressed weights: OK ({len(archive2)} bytes)")

    # 9. compress_byte_stream / decompress_byte_stream round-trip
    test_data = bytes(range(256)) * 4 + b"\x00" * 100 + b"\xff" * 100
    compressed = compress_byte_stream(test_data)
    decompressed = decompress_byte_stream(compressed)
    assert decompressed == test_data, (
        f"byte stream round-trip mismatch: {len(test_data)} -> {len(compressed)} -> {len(decompressed)}"
    )
    print(f"  byte stream compress: OK ({len(test_data)} -> {len(compressed)} bytes, "
          f"{len(compressed)/len(test_data)*100:.1f}%)")

    # 10. compress_byte_stream edge case: empty input
    assert decompress_byte_stream(compress_byte_stream(b"")) == b""
    print("  byte stream empty: OK")

    print("entropy_archive: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
