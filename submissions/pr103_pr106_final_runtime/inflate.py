#!/usr/bin/env python
"""Self-contained PR103 arithmetic decoder inside a PR106 packed runtime.

This final-runtime adapter is intentionally byte-bound to the candidate archive
proved in ``experiments/results/pr103_repack_pr106_standalone_20260507``.  It
loads no TAC modules and no scorer code at inflate time: the PR103 section
closure is embedded below, then the decoded PR106 HNeRV state/latents are run
through the vendored model definition.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import brotli  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised in clean runtime envs.
    brotli = None  # type: ignore[assignment]
    _BROTLI_IMPORT_ERROR = exc
else:
    _BROTLI_IMPORT_ERROR = None

try:
    import constriction  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised in clean runtime envs.
    constriction = None  # type: ignore[assignment]
    _CONSTRICTION_IMPORT_ERROR = exc
else:
    _CONSTRICTION_IMPORT_ERROR = None


CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600
LATENT_DIM = 28
BASE_CHANNELS = 36
EVAL_SIZE = (384, 512)

RUNTIME_FORMAT = "pr103_ac_decoder_inside_pr106_ff_packed_v1"
EXPECTED_PAYLOAD_SHA256 = "3272ec95a2ea5ec68feb1a53fa53f6b14bdae3883fac38ee2261cdadb1b16357"
RUNTIME_CLOSURE: dict[str, Any] = {
    "schema_version": 1,
    "format": RUNTIME_FORMAT,
    "section_lengths": {
        "br": 7192,
        "hists": 989,
        "merged_ac": 161380,
        "hi_hist": 0,
        "ac_fallback": 0,
    },
    "ac_fallback_set": [],
    "n_latent_hi_symbols": 0,
    "decoder_section_bytes": 169617,
    "decoder_section_sha256": "854278d7bb049a59b44a0fa85cbb849752ba84f02fbd7d91480c1a1ffcac42e5",
    "latents_section_bytes": 15849,
    "latents_section_sha256": "94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32",
    "brotli_quality": 11,
    "adaptive_lgwin": True,
    "ac_auto_fallback": True,
}

FIXED_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)
DECODER_STORAGE_ORDER = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)
DECODER_STREAM_ENDS = (1, 2, 22, 23, 26, 27, 28)
CONV4_STORAGE_PERMS = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}
CONV4_INVERSE_PERMS = {
    idx: tuple(int(value) for value in np.argsort(perm))
    for idx, perm in CONV4_STORAGE_PERMS.items()
}
DECODER_BYTE_MAPS = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}
AC_TENSOR_INDICES = (0, 2, 4, 6, 8, 10, 12, 21)
AC_SYMBOL_OFFSET = 128


class RuntimeClosureError(ValueError):
    """Raised when the PR103-on-PR106 runtime packet is not byte-faithful."""


class HNeRVDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        base_channels: int = BASE_CHANNELS,
        eval_size: tuple[int, int] = EVAL_SIZE,
    ) -> None:
        super().__init__()
        self.eval_size = eval_size
        self.base_h, self.base_w = 6, 8
        c = base_channels
        self.channels = [c, c, c, int(c * 0.75), int(c * 0.58), int(c * 0.5), int(c * 0.5)]
        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):
            in_ch, out_ch = self.channels[i], self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.stem(z).view(b, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips, strict=True):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


def runtime_dependency_versions() -> dict[str, str]:
    missing: list[str] = []
    if _BROTLI_IMPORT_ERROR is not None:
        missing.append(f"brotli: {_BROTLI_IMPORT_ERROR}")
    if _CONSTRICTION_IMPORT_ERROR is not None:
        missing.append(f"constriction: {_CONSTRICTION_IMPORT_ERROR}")
    if missing:
        raise RuntimeClosureError("missing runtime dependencies: " + "; ".join(missing))
    if not hasattr(constriction.stream.queue, "RangeDecoder"):  # type: ignore[union-attr]
        raise RuntimeClosureError("constriction.stream.queue.RangeDecoder missing")
    if not hasattr(constriction.stream.model, "Categorical"):  # type: ignore[union-attr]
        raise RuntimeClosureError("constriction.stream.model.Categorical missing")
    versions: dict[str, str] = {}
    for package in ("brotli", "constriction", "numpy", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _make_categorical(weights: np.ndarray) -> object:
    p = weights.astype(np.float64)
    p = np.maximum(p, 1e-10)
    p = p / p.sum()
    return constriction.stream.model.Categorical(p, perfect=False)  # type: ignore[union-attr]


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def _decode_mapped_u8(values: np.ndarray, byte_map: str) -> np.ndarray:
    if byte_map == "zig":
        return _zigzag_decode_u8(values)
    if byte_map == "negzig":
        return (-_zigzag_decode_u8(values).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (values.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return values.view(np.int8)
    raise RuntimeClosureError(f"unknown decoder byte map: {byte_map}")


def _split_pr106_packed_payload(payload: bytes) -> tuple[bytes, bytes]:
    if _sha256_bytes(payload) != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeClosureError("candidate payload SHA-256 mismatch")
    if len(payload) < 4:
        raise RuntimeClosureError("payload too short for PR106 packed header")
    if payload[0] != 0xFF:
        raise RuntimeClosureError(f"expected PR106 packed magic 0xff, got 0x{payload[0]:02x}")
    decoder_len = int.from_bytes(payload[1:4], "little")
    if decoder_len <= 0 or 4 + decoder_len >= len(payload):
        raise RuntimeClosureError(f"invalid PR106 decoder section length {decoder_len}")
    decoder = payload[4:4 + decoder_len]
    latents = payload[4 + decoder_len:]
    _validate_closure_sections(decoder, latents)
    return decoder, latents


def _validate_closure_sections(decoder: bytes, latents: bytes) -> None:
    closure = RUNTIME_CLOSURE
    section_lengths = closure["section_lengths"]
    expected_decoder_len = len(FIXED_STATE_SCHEMA) * 2 + sum(int(v) for v in section_lengths.values())
    if expected_decoder_len != len(decoder):
        raise RuntimeClosureError(
            f"section_lengths sum ({expected_decoder_len}) != decoder len ({len(decoder)})"
        )
    if int(closure["decoder_section_bytes"]) != len(decoder):
        raise RuntimeClosureError("decoder byte length mismatch")
    if str(closure["decoder_section_sha256"]) != _sha256_bytes(decoder):
        raise RuntimeClosureError("decoder SHA-256 mismatch")
    if int(closure["latents_section_bytes"]) != len(latents):
        raise RuntimeClosureError("latents byte length mismatch")
    if str(closure["latents_section_sha256"]) != _sha256_bytes(latents):
        raise RuntimeClosureError("latents SHA-256 mismatch")
    fallback_len = int(section_lengths.get("ac_fallback", 0))
    fallback_set = tuple(int(item) for item in closure.get("ac_fallback_set", ()))
    if fallback_len > 0 and not fallback_set:
        raise RuntimeClosureError("ac_fallback section non-empty but ac_fallback_set is empty")
    if fallback_len == 0 and fallback_set:
        raise RuntimeClosureError("ac_fallback_set non-empty but ac_fallback length is zero")


def _decode_decoder_ac(blob: bytes) -> dict[str, torch.Tensor]:
    section_lengths = RUNTIME_CLOSURE["section_lengths"]
    fallback_set = {int(item) for item in RUNTIME_CLOSURE["ac_fallback_set"]}
    invalid_fallback = fallback_set - set(AC_TENSOR_INDICES)
    if invalid_fallback:
        raise RuntimeClosureError(f"invalid ac_fallback_set indices: {sorted(invalid_fallback)}")
    active_ac_indices = tuple(idx for idx in AC_TENSOR_INDICES if idx not in fallback_set)

    scales_len = len(FIXED_STATE_SCHEMA) * 2
    br_len = int(section_lengths["br"])
    hists_len = int(section_lengths["hists"])
    merged_ac_len = int(section_lengths["merged_ac"])
    hi_hist_len = int(section_lengths["hi_hist"])
    ac_fallback_len = int(section_lengths["ac_fallback"])
    expected = scales_len + br_len + hists_len + merged_ac_len + hi_hist_len + ac_fallback_len
    if expected != len(blob):
        raise RuntimeClosureError(f"section_lengths sum ({expected}) != blob len ({len(blob)})")

    offset = 0
    scales_b = blob[offset:offset + scales_len]
    offset += scales_len
    br_b = blob[offset:offset + br_len]
    offset += br_len
    hists_b = blob[offset:offset + hists_len]
    offset += hists_len
    merged_ac = blob[offset:offset + merged_ac_len]
    offset += merged_ac_len
    hi_hist_b = blob[offset:offset + hi_hist_len]
    offset += hi_hist_len
    ac_fallback_b = blob[offset:offset + ac_fallback_len]
    offset += ac_fallback_len
    if offset != len(blob):
        raise RuntimeClosureError("decoder section slicing failed")

    fp16_scales = np.frombuffer(scales_b, dtype=np.float16)
    if len(fp16_scales) != len(FIXED_STATE_SCHEMA):
        raise RuntimeClosureError("fp16 scale table length mismatch")

    n_active = len(active_ac_indices)
    if hists_len > 0:
        hists_raw = brotli.decompress(hists_b)  # type: ignore[union-attr]
        if len(hists_raw) != n_active * 256:
            raise RuntimeClosureError(
                f"histogram raw length {len(hists_raw)} != expected {n_active * 256}"
            )
        hists = np.frombuffer(hists_raw, dtype=np.uint8).reshape(n_active, 256)
    else:
        hists = np.zeros((n_active, 256), dtype=np.uint8)

    if hi_hist_len > 0:
        hi_hist = np.frombuffer(brotli.decompress(hi_hist_b), dtype=np.uint16)  # type: ignore[union-attr]
    else:
        hi_hist = np.zeros(0, dtype=np.uint16)

    ac_arrays: dict[int, np.ndarray] = {}
    if merged_ac_len > 0:
        if merged_ac_len % 4 != 0:
            raise RuntimeClosureError(f"merged_ac len not 4-aligned: {merged_ac_len}")
        dec = constriction.stream.queue.RangeDecoder(np.frombuffer(merged_ac, dtype=np.uint32))  # type: ignore[union-attr]
        for k, idx in enumerate(active_ac_indices):
            cat = _make_categorical(hists[k])
            count = int(np.prod(FIXED_STATE_SCHEMA[idx][1]))
            arr = np.zeros(count, dtype=np.int32)
            for i in range(count):
                arr[i] = dec.decode(cat)
            ac_arrays[idx] = (arr - AC_SYMBOL_OFFSET).astype(np.int8).reshape(FIXED_STATE_SCHEMA[idx][1])
        n_latent_hi_symbols = int(RUNTIME_CLOSURE["n_latent_hi_symbols"])
        if n_latent_hi_symbols > 0:
            hi_cat = _make_categorical(hi_hist)
            for _ in range(n_latent_hi_symbols):
                dec.decode(hi_cat)

    if ac_fallback_len > 0:
        if not fallback_set:
            raise RuntimeClosureError("ac_fallback bytes present without ac_fallback_set")
        fallback_raw = brotli.decompress(ac_fallback_b)  # type: ignore[union-attr]
        fb_pos = 0
        for idx in AC_TENSOR_INDICES:
            if idx not in fallback_set:
                continue
            shape = FIXED_STATE_SCHEMA[idx][1]
            n_el = int(np.prod(shape))
            chunk = fallback_raw[fb_pos:fb_pos + n_el]
            fb_pos += n_el
            if len(chunk) != n_el:
                raise RuntimeClosureError(f"ac_fallback truncated at idx={idx}")
            u8 = np.frombuffer(chunk, dtype=np.uint8)
            ac_arrays[idx] = (u8.astype(np.int16) - AC_SYMBOL_OFFSET).astype(np.int8).reshape(shape)
        if fb_pos != len(fallback_raw):
            raise RuntimeClosureError("ac_fallback raw bytes had trailing data")
    elif fallback_set:
        raise RuntimeClosureError("ac_fallback_set present without ac_fallback bytes")

    ac_set = set(AC_TENSOR_INDICES)
    n_streams = sum(
        1
        for window_idx, end in enumerate(DECODER_STREAM_ENDS)
        if any(
            DECODER_STORAGE_ORDER[pos] not in ac_set
            for pos in range(0 if window_idx == 0 else DECODER_STREAM_ENDS[window_idx - 1], end)
        )
    )
    outputs: list[bytes] = []
    pos = 0
    for _ in range(n_streams):
        decompressor = brotli.Decompressor()  # type: ignore[union-attr]
        chunks: list[bytes] = []
        while pos < len(br_b) and not decompressor.is_finished():
            chunks.append(decompressor.process(br_b[pos:pos + 1]))
            pos += 1
        if not decompressor.is_finished():
            raise RuntimeClosureError("truncated non-AC brotli payload")
        outputs.append(b"".join(chunks))
    if pos != len(br_b):
        raise RuntimeClosureError("trailing non-AC brotli payload bytes")
    non_ac_concat = b"".join(outputs)

    state_dict: dict[str, torch.Tensor] = {}
    pos = 0
    for storage_idx in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[storage_idx]
        if storage_idx in ac_set:
            if storage_idx not in ac_arrays:
                raise RuntimeClosureError(f"missing AC tensor stream for {name}")
            q_i8 = ac_arrays[storage_idx]
        else:
            n_el = int(np.prod(shape))
            if pos + n_el > len(non_ac_concat):
                raise RuntimeClosureError(f"non-AC stream truncated at {name}")
            mapped = np.frombuffer(non_ac_concat[pos:pos + n_el], dtype=np.uint8)
            pos += n_el
            q_i8 = _decode_mapped_u8(mapped, DECODER_BYTE_MAPS.get(storage_idx, "zig"))
            if len(shape) == 4:
                storage_perm = CONV4_STORAGE_PERMS[storage_idx]
                inverse_perm = CONV4_INVERSE_PERMS[storage_idx]
                stored_shape = tuple(shape[index] for index in storage_perm)
                q_i8 = np.transpose(q_i8.reshape(stored_shape), inverse_perm).copy()
            else:
                q_i8 = q_i8.reshape(shape)
        scale = float(fp16_scales[storage_idx])
        state_dict[name] = torch.from_numpy(q_i8.astype(np.float32)) * scale
    if pos != len(non_ac_concat):
        raise RuntimeClosureError("non-AC stream had trailing decoded bytes")
    return state_dict


def _decode_pr106_fixed_latents(payload: bytes) -> torch.Tensor:
    raw = brotli.decompress(payload)  # type: ignore[union-attr]
    total = N_PAIRS * LATENT_DIM
    meta_len = LATENT_DIM * 4
    expected = total + meta_len + total
    if len(raw) != expected:
        raise RuntimeClosureError(f"bad PR106 fixed-latents len {len(raw)} != {expected}")
    lo = np.frombuffer(raw[:total], dtype=np.uint8).astype(np.uint16)
    mins = torch.from_numpy(np.frombuffer(raw[total:total + LATENT_DIM * 2], dtype=np.float16).copy()).float()
    scales = torch.from_numpy(
        np.frombuffer(raw[total + LATENT_DIM * 2:total + meta_len], dtype=np.float16).copy()
    ).float()
    hi = np.frombuffer(raw[total + meta_len:total + meta_len + total], dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(N_PAIRS, LATENT_DIM)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, N_PAIRS):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales.unsqueeze(0) + mins.unsqueeze(0)


def parse_pr103_pr106_archive(archive_bytes: bytes) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Decode candidate bytes to PR106 HNeRV state, latents, and metadata."""
    runtime_dependency_versions()
    decoder, latents_brotli = _split_pr106_packed_payload(archive_bytes)
    state_dict = _decode_decoder_ac(decoder)
    latents = _decode_pr106_fixed_latents(latents_brotli)
    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    return state_dict, latents, meta


def select_inflate_device() -> torch.device:
    """Return the best available inflate device for this runtime.

    CUDA remains the contest-promotion path for the existing active floor.
    CPU is required for the public GHA axis, so absence of CUDA is not a
    runtime-contract failure. The resulting runtime tree must be re-evaluated
    on both axes before any score promotion.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inflate(src_bin: str, dst_raw: str) -> int:
    versions = runtime_dependency_versions()
    print(
        "[pr103-pr106-final] deps "
        + " ".join(f"{name}={version}" for name, version in sorted(versions.items())),
        file=sys.stderr,
    )
    decoder_sd, latents, meta = parse_pr103_pr106_archive(Path(src_bin).read_bytes())

    device = select_inflate_device()
    print(f"[pr103-pr106-final] inflate_device={device.type}", file=sys.stderr)
    decoder = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = int(meta["n_pairs"])
    eval_h, eval_w = meta["eval_size"]
    n_frames = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, int(eval_h), int(eval_w))
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
            frames = up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            fout.write(frames.tobytes())
            n_frames += batch * 2
    print(f"saved {n_frames} frames")
    return n_frames


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--dependency-check":
        versions = runtime_dependency_versions()
        print(
            "[pr103-pr106-final] dependency check "
            + " ".join(f"{name}={version}" for name, version in sorted(versions.items()))
        )
        raise SystemExit(0)
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python -m submissions.pr103_pr106_final_runtime.inflate "
            "<src.bin> <dst.raw>"
        )
    inflate(sys.argv[1], sys.argv[2])
