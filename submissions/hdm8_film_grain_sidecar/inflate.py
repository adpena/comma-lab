#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Inflate an HDM8 HNeRV archive with an optional deterministic postfilter.

This runtime consumes the same PR106/HDM8 archive shape as
``submissions/pr106_latent_sidecar_r2_pr101_grammar``:

  format_id=0x01 — legacy brotli-compressed (dim u8, delta_q i8) sidecar
  format_id=0x02 — PR101 ranked-Huffman/no-op grammar sidecar (this variant's
                    primary encoding; saves 42 bytes net vs format_id=0x01)
  format_id=0x03 — format_id=0x02 plus an archive-packed postfilter selector
                    JSON trailer; selector bytes are charged in archive.zip
  format_id=0x04 — format_id=0x03 but the selector JSON trailer is
                    brotli-compressed before being stored in archive.zip

Both format_ids reconstruct the (dims, delta_q) arrays bit-identical, which is
parser/decoder-consumption evidence only. Score components require exact
auth-eval evidence under the scored runtime.

Reads <src>.bin, reconstructs PR106 state_dict + latents, applies the per-pair
(dim, delta) corrections, runs the HNeRV decoder at 384x512, bicubic-upsamples
to camera resolution (874x1164), applies the fixed postfilter mode in
``postfilter_config.json``, rounds to uint8, and writes contiguous
(N, H, W, 3) bytes to <dst>.

The postfilter path is scorer-free and deterministic. It intentionally keeps
mode selection in a runtime config file instead of an environment variable so
an exact-eval packet has one auditable behaviour.

Invoked by inflate.sh as:
    python -m submissions.hdm8_film_grain_sidecar.inflate <src.bin> <dst.raw>
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_packed_archive  # type: ignore[import-not-found]
from model import HNeRVDecoder  # type: ignore[import-not-found]
from pr101_grammar import RankedSidecarSchema, decode_ranked_no_op_sidecar  # type: ignore[import-not-found]


CAMERA_H, CAMERA_W = 874, 1164
POSTFILTER_CONFIG_PATH = HERE / "postfilter_config.json"
POSTFILTER_CONFIG_SCHEMA = "hdm8_film_grain_sidecar_postfilter_config_v1"
SIDECAR_MAGIC = 0xFE
SIDECAR_FORMAT_BROTLI = 0x01
SIDECAR_FORMAT_PR101_GRAMMAR = 0x02
SIDECAR_FORMAT_PR101_SELECTOR = 0x03
SIDECAR_FORMAT_PR101_SELECTOR_BROTLI = 0x04
SUPPORTED_FORMATS = (
    SIDECAR_FORMAT_BROTLI,
    SIDECAR_FORMAT_PR101_GRAMMAR,
    SIDECAR_FORMAT_PR101_SELECTOR,
    SIDECAR_FORMAT_PR101_SELECTOR_BROTLI,
)
DELTA_SCALE = 0.01
NO_OP_DIM = 255
DEFAULT_BATCH_PAIRS = 16
UNSHARP_ROW = torch.tensor([1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0])

# PR101-grammar schema for this variant: n_pairs=600, n_dims=28,
# deltas=(-2,-1,1,2), huff_min/max=(2,8), no-op sentinel=255.
PR101_SCHEMA = RankedSidecarSchema(
    n_pairs=600,
    n_dims=28,
    deltas=(-2, -1, 1, 2),
    huff_min_len=2,
    huff_max_len=8,
    no_op_sentinel=255,
)


def parse_sidecar_archive_with_selector(
    bin_bytes: bytes,
) -> tuple[int, bytes, bytes, bytes | None, dict[str, object] | None]:
    """Slice apart the wrapper and dispatch on format_id.

    Returns ``(format_id, pr106_bytes, sidecar_blob, framing_meta, selector_config)``.

    For format_id=0x01 (brotli): ``framing_meta`` is ``None``; ``sidecar_blob``
    is the brotli-compressed payload.

    For format_id=0x02 (PR101 grammar): ``framing_meta`` is the bytes
    immediately after the PR101 payload, namely
    ``noop_count(2) | dim_bytes(2) | rank_bytes(1) | noop_rank_bytes(1)``;
    ``sidecar_blob`` is the PR101 grammar payload of length pr101_payload_len.

    For format_id=0x03: same as 0x02, followed by
    ``selector_json_len(2) | selector_json``. For format_id=0x04 the selector
    payload is brotli-compressed JSON. The selector config is video side
    information and must be present in archive bytes, not only runtime config,
    for a byte-closed packet.
    """
    if not bin_bytes:
        raise ValueError("empty archive")
    if bin_bytes[0] != SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{bin_bytes[0]:02X}, expected 0x{SIDECAR_MAGIC:02X}"
        )
    format_id = bin_bytes[1]
    if format_id not in SUPPORTED_FORMATS:
        raise ValueError(
            f"sidecar format_id 0x{format_id:02X} not supported; expected one of "
            f"{', '.join(f'0x{f:02X}' for f in SUPPORTED_FORMATS)}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    pr106_bytes = bin_bytes[pos : pos + pr106_len]
    pos += pr106_len

    if format_id == SIDECAR_FORMAT_BROTLI:
        if pos + 2 > len(bin_bytes):
            raise ValueError("sidecar archive truncated before sidecar_len")
        (sidecar_len,) = struct.unpack_from("<H", bin_bytes, pos)
        pos += 2
        sidecar_blob = bin_bytes[pos : pos + sidecar_len]
        pos += sidecar_len
        if pos != len(bin_bytes):
            raise ValueError(
                f"sidecar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
            )
        return format_id, pr106_bytes, sidecar_blob, None, None

    # format_id in {SIDECAR_FORMAT_PR101_GRAMMAR, SIDECAR_FORMAT_PR101_SELECTOR*}
    if pos + 2 > len(bin_bytes):
        raise ValueError("pr101_grammar archive truncated before pr101_payload_len")
    (pr101_payload_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    pr101_payload = bin_bytes[pos : pos + pr101_payload_len]
    pos += pr101_payload_len
    if pos + 6 > len(bin_bytes):
        raise ValueError("pr101_grammar archive truncated before framing meta")
    framing_meta = bin_bytes[pos : pos + 6]
    pos += 6
    selector_config: dict[str, object] | None = None
    if format_id in {SIDECAR_FORMAT_PR101_SELECTOR, SIDECAR_FORMAT_PR101_SELECTOR_BROTLI}:
        if pos + 2 > len(bin_bytes):
            raise ValueError("selector archive truncated before selector_json_len")
        (selector_len,) = struct.unpack_from("<H", bin_bytes, pos)
        pos += 2
        selector_bytes = bin_bytes[pos : pos + selector_len]
        pos += selector_len
        if len(selector_bytes) != selector_len:
            raise ValueError("selector archive truncated in selector_json payload")
        if format_id == SIDECAR_FORMAT_PR101_SELECTOR_BROTLI:
            selector_bytes = brotli.decompress(selector_bytes)
        selector_config = json.loads(selector_bytes.decode("utf-8"))
        # Fail closed on malformed embedded selector payloads before decoding frames.
        validate_postfilter_config_payload(selector_config)
    if pos != len(bin_bytes):
        raise ValueError(
            f"pr101_grammar archive trailing bytes: pos={pos} vs total={len(bin_bytes)}"
        )
    return format_id, pr106_bytes, pr101_payload, framing_meta, selector_config


def parse_sidecar_archive(bin_bytes: bytes) -> tuple[int, bytes, bytes, bytes | None]:
    """Backward-compatible four-field parser for tests and parser-consumption probes."""
    format_id, pr106_bytes, sidecar_blob, framing_meta, _selector_config = (
        parse_sidecar_archive_with_selector(bin_bytes)
    )
    return format_id, pr106_bytes, sidecar_blob, framing_meta


def decode_brotli_sidecar(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode the legacy format_id=0x01 brotli-compressed sidecar payload."""
    raw = brotli.decompress(blob)
    n = struct.unpack_from("<H", raw, 0)[0]
    arr = np.frombuffer(raw[2 : 2 + 2 * n], dtype=np.uint8).reshape(n, 2)
    dim = arr[:, 0]
    delta_q = arr[:, 1].view(np.int8)
    return dim, delta_q


def decode_pr101_grammar_sidecar(
    payload: bytes, framing_meta: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the format_id=0x02 PR101 ranked-Huffman/no-op grammar sidecar."""
    noop_count, dim_bytes, rank_bytes, noop_rank_bytes = struct.unpack("<HHBB", framing_meta)
    dims, delta_indices = decode_ranked_no_op_sidecar(
        payload,
        schema=PR101_SCHEMA,
        dim_bytes=int(dim_bytes),
        rank_bytes=int(rank_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
        noop_count=int(noop_count),
    )
    # Reconstruct (dim, delta_q) arrays matching format_id=0x01 byte semantics.
    dim_arr = dims.astype(np.int64)
    # Convert no-op sentinel from schema (255) to NO_OP_DIM (also 255).
    dim_arr_u8 = np.where(dim_arr == PR101_SCHEMA.no_op_sentinel, NO_OP_DIM, dim_arr).astype(np.uint8)
    delta_lookup = np.array(PR101_SCHEMA.deltas, dtype=np.int8)
    delta_q_arr = np.zeros(PR101_SCHEMA.n_pairs, dtype=np.int8)
    valid_mask = dim_arr != PR101_SCHEMA.no_op_sentinel
    delta_q_arr[valid_mask] = delta_lookup[delta_indices[valid_mask]]
    return dim_arr_u8, delta_q_arr


def apply_sidecar_corrections(
    latents: torch.Tensor,
    dim_arr: np.ndarray,
    delta_q_arr: np.ndarray,
    *,
    scale: float = DELTA_SCALE,
) -> torch.Tensor:
    """In-place add per-pair correction to (n, latent_dim) latents tensor."""
    n = latents.shape[0]
    for p in range(n):
        d = int(dim_arr[p])
        if d == NO_OP_DIM:
            continue
        latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * scale
    return latents


def select_inflate_device() -> torch.device:
    """Select an auth-eval-safe inflate device (cuda or cpu; MPS forbidden)."""
    policy = os.environ.get("PACT_INFLATE_DEVICE", "auto").strip().lower()
    if policy in {"mps", "metal"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE=mps is forbidden for auth-eval inflate; use cpu or cuda"
        )
    if policy not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(
            "PACT_INFLATE_DEVICE must be one of auto, cpu, cuda "
            f"(got {policy!r})"
        )
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_batch_pairs() -> int:
    """Return the deterministic decoder batch size for pair forwards."""
    raw = os.environ.get("PACT_INFLATE_BATCH_PAIRS")
    if raw is None:
        return DEFAULT_BATCH_PAIRS
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {raw!r})"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"PACT_INFLATE_BATCH_PAIRS must be a positive integer (got {value})"
        )
    return value


def parse_postfilter_mode(mode: str) -> tuple[str, float]:
    """Return ``(name, value)`` for deterministic postfilter mode strings."""
    text = str(mode).strip()
    if not text:
        raise ValueError("postfilter mode cannot be empty")
    if ":" not in text:
        return text, 0.0
    name, raw = text.split(":", 1)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"postfilter mode has non-float value: {mode!r}") from exc
    return name.strip(), value


def _parse_rgb_triplet(raw: str) -> tuple[float, float, float]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError(f"expected three comma-separated RGB values, got {raw!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def validate_postfilter_mode(mode: str) -> str:
    """Validate a postfilter mode without mutating any frames."""
    if "+" in mode:
        parts = [part.strip() for part in mode.split("+")]
        if not all(parts):
            raise ValueError(f"empty sub-mode in composite postfilter mode {mode!r}")
        for part in parts:
            validate_postfilter_mode(part)
        return mode
    if mode.startswith("even_") or mode.startswith("odd_"):
        validate_postfilter_mode(mode.split("_", 1)[1])
        return mode
    if mode.startswith("translate:"):
        _name, raw = mode.split(":", 1)
        dy_raw, dx_raw = raw.split(",", 1)
        int(dy_raw)
        int(dx_raw)
        return mode
    if mode.startswith("rgb_bias:"):
        _name, raw = mode.split(":", 1)
        _parse_rgb_triplet(raw)
        return mode
    if mode.startswith("rgb_scale:"):
        _name, raw = mode.split(":", 1)
        _parse_rgb_triplet(raw)
        return mode
    name, _value = parse_postfilter_mode(mode)
    supported = {
        "none",
        "bias",
        "contrast",
        "gamma",
        "unsharp",
        "soften",
        "adaptive",
        "grain",
        "grain_luma",
        "grain_chroma",
        "grain_var",
        "blue",
        "checker",
        "tile_chroma",
    }
    if name not in supported:
        raise ValueError(f"unsupported postfilter mode {mode!r}")
    return mode


def validate_postfilter_config_payload(payload: dict[str, object]) -> dict[str, object]:
    """Validate and normalize a postfilter config payload."""
    schema = payload.get("schema")
    if schema != POSTFILTER_CONFIG_SCHEMA:
        raise ValueError(
            f"bad postfilter config schema: {schema!r}; expected {POSTFILTER_CONFIG_SCHEMA!r}"
        )
    mode = str(payload.get("mode", "none")).strip()
    if mode != "selector":
        return {"schema": POSTFILTER_CONFIG_SCHEMA, "mode": validate_postfilter_mode(mode)}

    palette_raw = payload.get("palette")
    if not isinstance(palette_raw, list) or not palette_raw:
        raise ValueError("selector postfilter config requires non-empty palette list")
    palette = [validate_postfilter_mode(str(item).strip()) for item in palette_raw]

    indices_raw = payload.get("selector_indices")
    if not isinstance(indices_raw, list) or not indices_raw:
        raise ValueError("selector postfilter config requires non-empty selector_indices list")
    indices: list[int] = []
    for raw_idx in indices_raw:
        idx = int(raw_idx)
        if idx < 0 or idx >= len(palette):
            raise ValueError(
                f"selector index {idx} out of range for palette size {len(palette)}"
            )
        indices.append(idx)
    return {
        "schema": POSTFILTER_CONFIG_SCHEMA,
        "mode": "selector",
        "palette": palette,
        "selector_indices": indices,
    }


def load_postfilter_config(config_path: Path = POSTFILTER_CONFIG_PATH) -> dict[str, object]:
    """Read and validate the fixed postfilter config from the runtime packet."""
    if not config_path.exists():
        return {"schema": POSTFILTER_CONFIG_SCHEMA, "mode": "none"}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_postfilter_config_payload(payload)


def load_postfilter_mode(config_path: Path = POSTFILTER_CONFIG_PATH) -> str:
    """Read the fixed postfilter mode from the runtime packet config."""
    return str(load_postfilter_config(config_path).get("mode", "none"))


def _postfilter_kernel(device: torch.device) -> torch.Tensor:
    row = UNSHARP_ROW.to(device)
    kernel_2d = torch.outer(row, row) / (row.sum() ** 2)
    return kernel_2d.expand(3, 1, 9, 9).contiguous()


def _translate_keep_borders(frames_bchw: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    if dy == 0 and dx == 0:
        return frames_bchw
    out = frames_bchw.clone()
    _, _, height, width = frames_bchw.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = frames_bchw[
            :, :, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def _coordinate_noise(
    frames_bchw: torch.Tensor,
    *,
    frame_start: int,
    channel_independent: bool,
) -> torch.Tensor:
    batch, channels, height, width = frames_bchw.shape
    yy = torch.arange(height, device=frames_bchw.device, dtype=torch.float32).view(
        1, 1, height, 1
    )
    xx = torch.arange(width, device=frames_bchw.device, dtype=torch.float32).view(
        1, 1, 1, width
    )
    ff = torch.arange(
        frame_start,
        frame_start + batch,
        device=frames_bchw.device,
        dtype=torch.float32,
    ).view(batch, 1, 1, 1)
    cc = torch.arange(channels, device=frames_bchw.device, dtype=torch.float32).view(
        1, channels, 1, 1
    )
    if not channel_independent:
        cc = torch.zeros_like(cc)
    phase = xx * 12.9898 + yy * 78.233 + ff * 37.719 + cc * 19.191
    return (torch.frac(torch.sin(phase) * 43758.5453) - 0.5) * 2.0


def _apply_grain(frames_bchw: torch.Tensor, mode: str, *, frame_start: int) -> torch.Tensor:
    name, value = parse_postfilter_mode(mode)
    amp = float(value)
    if name == "checker":
        batch, channels, height, width = frames_bchw.shape
        yy = torch.arange(height, device=frames_bchw.device).view(1, 1, height, 1)
        xx = torch.arange(width, device=frames_bchw.device).view(1, 1, 1, width)
        ff = torch.arange(frame_start, frame_start + batch, device=frames_bchw.device).view(
            batch, 1, 1, 1
        )
        pattern = (((xx + yy + ff) & 1).float() * 2.0 - 1.0).expand(
            batch, channels, height, width
        )
        return frames_bchw + amp * pattern
    if name == "tile_chroma":
        batch, _channels, height, width = frames_bchw.shape
        base = torch.tensor(
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
            dtype=frames_bchw.dtype,
            device=frames_bchw.device,
        )
        reps_h = (height + 7) // 8
        reps_w = (width + 7) // 8
        pattern = base.repeat(reps_h, reps_w)[:height, :width].view(1, 1, height, width)
        pattern = pattern.expand(batch, 1, height, width)
        out = frames_bchw.clone()
        out[:, 0:1].add_(amp * pattern)
        out[:, 2:3].sub_(amp * pattern)
        return out

    if name == "grain":
        return frames_bchw + amp * _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
    if name == "grain_luma":
        noise = _coordinate_noise(
            frames_bchw[:, :1],
            frame_start=frame_start,
            channel_independent=False,
        )
        return frames_bchw + amp * noise.expand_as(frames_bchw)
    if name == "grain_chroma":
        noise = _coordinate_noise(
            frames_bchw[:, :1],
            frame_start=frame_start,
            channel_independent=False,
        )
        weights = torch.tensor([1.0, -0.5, -0.5], device=frames_bchw.device).view(
            1, 3, 1, 1
        )
        return frames_bchw + amp * noise.expand_as(frames_bchw) * weights
    if name == "blue":
        noise = _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
        low = F.avg_pool2d(F.pad(noise, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        high = noise - low
        high = high / high.flatten(1).std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return frames_bchw + amp * high.clamp(-2.0, 2.0)
    if name == "grain_var":
        noise = _coordinate_noise(
            frames_bchw,
            frame_start=frame_start,
            channel_independent=True,
        )
        luma = (
            0.299 * frames_bchw[:, 0:1]
            + 0.587 * frames_bchw[:, 1:2]
            + 0.114 * frames_bchw[:, 2:3]
        )
        local_mean = F.avg_pool2d(F.pad(luma, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_sq = F.avg_pool2d(F.pad(luma.square(), (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_var = (local_sq - local_mean.square()).clamp_min(0)
        scale = (local_var / (local_var + 64.0)).expand_as(frames_bchw)
        return frames_bchw + amp * noise * scale
    raise ValueError(f"unsupported grain mode {mode!r}")


def apply_postfilter(
    frames_bchw: torch.Tensor,
    mode: str,
    *,
    frame_start: int,
) -> torch.Tensor:
    """Apply the deterministic postfilter before clamp/round/write."""
    if "+" in mode:
        out = frames_bchw
        for part in mode.split("+"):
            out = apply_postfilter(out, part.strip(), frame_start=frame_start)
        return out
    if mode.startswith("even_") or mode.startswith("odd_"):
        want_even = mode.startswith("even_")
        inner_mode = mode.split("_", 1)[1]
        filtered = apply_postfilter(frames_bchw, inner_mode, frame_start=frame_start)
        frame_ids = torch.arange(
            frame_start,
            frame_start + frames_bchw.shape[0],
            device=frames_bchw.device,
        )
        parity_mask = ((frame_ids % 2) == 0) if want_even else ((frame_ids % 2) == 1)
        return torch.where(parity_mask.view(-1, 1, 1, 1), filtered, frames_bchw)
    if mode.startswith("translate:"):
        _name, raw = mode.split(":", 1)
        dy_raw, dx_raw = raw.split(",", 1)
        return _translate_keep_borders(frames_bchw, dy=int(dy_raw), dx=int(dx_raw))
    if mode.startswith("rgb_bias:"):
        _name, raw = mode.split(":", 1)
        dr, dg, db = _parse_rgb_triplet(raw)
        delta = torch.tensor([dr, dg, db], device=frames_bchw.device).view(1, 3, 1, 1)
        return frames_bchw + delta
    if mode.startswith("rgb_scale:"):
        _name, raw = mode.split(":", 1)
        sr, sg, sb = _parse_rgb_triplet(raw)
        scale = torch.tensor([sr, sg, sb], device=frames_bchw.device).view(1, 3, 1, 1)
        return frames_bchw * scale

    name, value = parse_postfilter_mode(mode)
    if name == "none":
        return frames_bchw
    if name == "bias":
        return frames_bchw + value
    if name == "contrast":
        return (frames_bchw - 127.5) * (1.0 + value) + 127.5
    if name == "gamma":
        return (frames_bchw / 255.0).clamp(0.0, 1.0).pow(value) * 255.0
    if name in {"grain", "grain_luma", "grain_chroma", "grain_var", "blue", "checker", "tile_chroma"}:
        return _apply_grain(frames_bchw, mode, frame_start=frame_start)

    kernel = _postfilter_kernel(frames_bchw.device)
    padded = F.pad(frames_bchw, (4, 4, 4, 4), mode="reflect")
    blur = F.conv2d(padded, kernel, padding=0, groups=3)
    detail = frames_bchw - blur

    if name == "unsharp":
        return frames_bchw + value * detail
    if name == "soften":
        return frames_bchw - value * detail
    if name == "adaptive":
        luma = (
            0.299 * frames_bchw[:, 0:1]
            + 0.587 * frames_bchw[:, 1:2]
            + 0.114 * frames_bchw[:, 2:3]
        )
        local_mean = F.avg_pool2d(F.pad(luma, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_sq = F.avg_pool2d(F.pad(luma.square(), (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        local_var = (local_sq - local_mean.square()).clamp_min(0)
        alpha = value * (local_var / (local_var + 100.0))
        return frames_bchw + alpha * detail
    raise ValueError(f"unsupported postfilter mode {mode!r}")


def _mode_for_pair(config: dict[str, object], pair_index: int) -> str:
    mode = str(config.get("mode", "none"))
    if mode != "selector":
        return mode
    palette = config["palette"]
    indices = config["selector_indices"]
    if not isinstance(palette, list) or not isinstance(indices, list):
        raise ValueError("bad selector config shape")
    if pair_index >= len(indices):
        raise ValueError(
            f"selector config has {len(indices)} entries, cannot decode pair {pair_index}"
        )
    return str(palette[int(indices[pair_index])])


def apply_postfilter_config(
    frames_bchw: torch.Tensor,
    config: dict[str, object],
    *,
    pair_start: int,
) -> torch.Tensor:
    """Apply either a fixed mode or a per-pair selector to a flat pair batch."""
    mode = str(config.get("mode", "none"))
    if mode != "selector":
        return apply_postfilter(frames_bchw, mode, frame_start=pair_start * 2)
    if frames_bchw.shape[0] % 2 != 0:
        raise ValueError("selector postfilter expects complete frame pairs")
    chunks: list[torch.Tensor] = []
    n_pairs = frames_bchw.shape[0] // 2
    for offset in range(n_pairs):
        pair_index = pair_start + offset
        pair_frames = frames_bchw[offset * 2 : offset * 2 + 2]
        pair_mode = _mode_for_pair(config, pair_index)
        chunks.append(apply_postfilter(pair_frames, pair_mode, frame_start=pair_index * 2))
    return torch.cat(chunks, dim=0)


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()

    format_id, pr106_bytes, sidecar_blob, framing_meta, archive_postfilter_config = (
        parse_sidecar_archive_with_selector(archive_bytes)
    )
    decoder_sd, latents, meta = parse_packed_archive(pr106_bytes)

    if format_id == SIDECAR_FORMAT_BROTLI:
        if sidecar_blob:
            dim_arr, delta_q_arr = decode_brotli_sidecar(sidecar_blob)
        else:
            dim_arr = np.full(PR101_SCHEMA.n_pairs, NO_OP_DIM, dtype=np.uint8)
            delta_q_arr = np.zeros(PR101_SCHEMA.n_pairs, dtype=np.int8)
    else:  # SIDECAR_FORMAT_PR101_GRAMMAR
        if framing_meta is None:
            raise ValueError("framing_meta missing for format_id=0x02 payload")
        dim_arr, delta_q_arr = decode_pr101_grammar_sidecar(sidecar_blob, framing_meta)

    n_corrections = int((dim_arr != NO_OP_DIM).sum())
    print(
        f"[inflate] format_id=0x{format_id:02X} sidecar applied: "
        f"{n_corrections}/{len(dim_arr)} pairs corrected",
        file=sys.stderr,
    )
    apply_sidecar_corrections(latents, dim_arr, delta_q_arr)

    try:
        device = select_inflate_device()
        batch_pairs = select_batch_pairs()
        postfilter_config = archive_postfilter_config or load_postfilter_config()
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    if str(postfilter_config.get("mode")) == "selector":
        indices = postfilter_config.get("selector_indices")
        if not isinstance(indices, list) or len(indices) < int(n_pairs):
            raise SystemExit(
                f"selector config has {0 if not isinstance(indices, list) else len(indices)} "
                f"entries; need at least {n_pairs}"
            )
    eval_h, eval_w = meta["eval_size"]
    postfilter_label = str(postfilter_config.get("mode", "none"))
    print(
        f"[inflate] PR106+sidecar: decoder loaded, device={device.type}, "
        f"batch_pairs={batch_pairs}, postfilter={postfilter_label}, "
        f"running {n_pairs} pair forwards...",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, batch_pairs):
            j = min(i + batch_pairs, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            up = apply_postfilter_config(up, postfilter_config, pair_start=i)
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python -m submissions.hdm8_film_grain_sidecar.inflate "
            "<src.bin> <dst.raw>"
        )
    inflate(sys.argv[1], sys.argv[2])
