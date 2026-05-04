"""Build a Score-Jacobian Karhunen-Loève residual side-info file.

Wave-Ω-1, Council #2 (FIELDS-MEDAL session 2026-05-01).

This script is the COMPRESS-TIME stage of the SJ-KL primitive:

  1. Loads the renderer's per-pair output for the decode slot that will be
     corrected at inflate time. Current robust_current runtime applies SJ-KL
     only to JointFrameGenerator pair slot 0 (`fake1`), so the production
     default is `--target-slot 0`.
  2. Loads the GT pixel pairs from upstream/videos/*.mkv.
  3. Computes the residual r[i] = GT_pair[i,target_slot] - renderer_output[i].
  4. On a representative subsample of frame pairs (default: 30 evenly
     spaced), computes the global SJ-KL basis using the public
     SegNet+PoseNet scorers.
  5. Encodes each per-pair residual as low-bit alpha coefficients.
  6. Packs (basis + alpha block) into a single sjkl.bin file ready
     to drop into an archive (e.g., as a sister to actuator.npz.br).

The DECODE-TIME counterpart (the bit that lives in the archive's inflate
script) is roughly:

    payload = open("sjkl.bin", "rb").read()
    basis = unpack_sjkl_basis(payload[:basis_len])
    block = unpack_alpha_block(payload[basis_len:])
    for i, (alpha_q, a_min, a_step) in enumerate(block):
        r_hat = decode_residual(alpha_q, a_min, a_step, basis)
        fake1[i] = fake1[i] + r_hat   # mirrors pr67_inflate.py:884

This script does NOT load any scorer at decode time — the basis is shipped
in the archive. Therefore the strict-scorer-rule (CLAUDE.md non-negotiable)
is satisfied.

DOES NOT dispatch any GPU job. Use this locally to build a candidate sjkl.bin
that an operator then bundles into a Vast.ai 4090 contest-CUDA eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Make src/ importable when run as a script
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO / "upstream") not in sys.path:
    sys.path.insert(0, str(_REPO / "upstream"))

from tac.sjkl_basis import (
    SJKL_MAGIC,
    SJKLBasis,
    compute_sjkl_basis,
    encode_residual,
    pack_sjkl_basis,
    unpack_sjkl_basis,
)


SJKL_BLOCK_MAGIC = b"SJKB"  # alpha-block magic
SJKL_BLOCK_V2_MAGIC = b"SJK2"  # sparse/bitpacked alpha-block magic


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def resolve_sjkl_build_device(device_arg: str, *, allow_non_cuda: bool) -> torch.device:
    """Resolve the build device with fail-closed CUDA defaults."""
    requested = torch.device(device_arg)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            if allow_non_cuda:
                print(
                    "[sjkl-basis] WARNING: --device cuda but CUDA unavailable; "
                    "falling back to CPU because --allow-non-cuda was set. "
                    "Result is ADVISORY only.",
                    file=sys.stderr,
                )
                return torch.device("cpu")
            raise RuntimeError(
                "--device cuda requested but CUDA is unavailable. Re-run on a "
                "CUDA host or pass --allow-non-cuda for an advisory build."
            )
        return requested
    if not allow_non_cuda:
        raise RuntimeError(
            f"--device {device_arg!r} is non-CUDA. Pass --allow-non-cuda only "
            "for advisory smoke builds; production SJ-KL bytes must be built "
            "on CUDA."
        )
    return requested


def _load_scorers(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load SegNet + PoseNet from upstream/models. CUDA strongly preferred
    (CLAUDE.md non-negotiable for any score-relevant computation).
    """
    if device.type == "cuda":
        pass  # OK
    elif device.type == "mps":
        print(
            "[sjkl-basis] WARNING: running on MPS — basis is ADVISORY only "
            "per CLAUDE.md 'MPS auth eval is NOISE'. Re-run on CUDA before "
            "shipping the resulting sjkl.bin in any submission archive."
        )
    else:
        print(f"[sjkl-basis] WARNING: device={device.type} — basis is ADVISORY only.")

    from modules import SegNet, PoseNet  # type: ignore  # upstream
    from safetensors.torch import load_file

    segnet = SegNet().eval().to(device)
    posenet = PoseNet().eval().to(device)
    seg_sd = load_file(str(_REPO / "upstream" / "models" / "segnet.safetensors"), device=str(device))
    pose_sd = load_file(str(_REPO / "upstream" / "models" / "posenet.safetensors"), device=str(device))
    segnet.load_state_dict(seg_sd)
    posenet.load_state_dict(pose_sd)
    return segnet, posenet


def _uint_dtype_for_bits(bits: int):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    raise ValueError(f"alpha_bits {bits} > 16 not supported")


def _quantize_alpha(alpha: np.ndarray, *, alpha_bits: int) -> tuple[np.ndarray, float, float]:
    if alpha_bits <= 0 or alpha_bits > 16:
        raise ValueError(f"alpha_bits must be in [1, 16], got {alpha_bits}")
    levels = (1 << alpha_bits) - 1
    alpha = alpha.astype(np.float32, copy=False)
    a_min = float(alpha.min())
    a_max = float(alpha.max())
    if a_max == a_min:
        return np.zeros_like(alpha, dtype=_uint_dtype_for_bits(alpha_bits)), a_min, 1.0
    a_step = (a_max - a_min) / levels
    a_q = np.clip(np.round((alpha - a_min) / a_step), 0, levels).astype(
        _uint_dtype_for_bits(alpha_bits)
    )
    return a_q, a_min, a_step


def _pack_bitpacked_uints(values: np.ndarray, *, bits: int) -> bytes:
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    if flat.size == 0:
        return b""
    max_value = (1 << bits) - 1
    if int(flat.max()) > max_value:
        raise ValueError(f"value exceeds {bits}-bit range")
    out = bytearray(math.ceil(flat.size * bits / 8))
    bit_pos = 0
    for value in flat.tolist():
        value = int(value)
        byte_idx = bit_pos // 8
        offset = bit_pos % 8
        out[byte_idx] |= (value << offset) & 0xFF
        spill = value >> (8 - offset)
        cursor = byte_idx + 1
        remaining = bits - (8 - offset)
        while remaining > 0:
            out[cursor] |= spill & 0xFF
            spill >>= 8
            remaining -= 8
            cursor += 1
        bit_pos += bits
    return bytes(out)


def _unpack_bitpacked_uints(payload: bytes, *, count: int, bits: int) -> np.ndarray:
    if bits <= 0 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    expected = math.ceil(count * bits / 8)
    if len(payload) != expected:
        raise ValueError(f"bitpacked payload length mismatch: expected {expected}, got {len(payload)}")
    out = np.zeros(count, dtype=_uint_dtype_for_bits(bits))
    bit_pos = 0
    mask = (1 << bits) - 1
    for idx in range(count):
        byte_idx = bit_pos // 8
        offset = bit_pos % 8
        window = 0
        for b in range(4):
            if byte_idx + b < len(payload):
                window |= payload[byte_idx + b] << (8 * b)
        out[idx] = (window >> offset) & mask
        bit_pos += bits
    return out


def pack_alpha_block(
    alpha_qs: list[np.ndarray],
    alpha_mins: list[float],
    alpha_steps: list[float],
    *,
    alpha_bits: int = 6,
    brotli_quality: int = 11,
    pair_indices: list[int] | None = None,
    sparse_bitpacked: bool = False,
) -> bytes:
    """Pack the per-pair alpha quantizations into a brotli-compressed block.

    Legacy layout (pre-brotli):
      MAGIC(4) | n_pairs(uint16) | k(uint16) | alpha_bits(uint8) |
        alpha_mins_fp16[n_pairs] | alpha_steps_fp16[n_pairs] |
        alpha_qs[n_pairs * k] (uint8 if alpha_bits<=8 else uint16, packed)

    Sparse layout (pre-brotli, ``sparse_bitpacked=True``):
      SJK2(4) | n_rows(uint16) | k(uint16) | alpha_bits(uint8) |
        pair_indices_uint16[n_rows] | alpha_mins_fp16[n_rows] |
        alpha_steps_fp16[n_rows] | bitpacked alpha_qs[n_rows * k]
    """
    import brotli

    n_pairs = len(alpha_qs)
    if n_pairs == 0:
        raise ValueError("empty alpha block")
    k = alpha_qs[0].shape[0]
    for a in alpha_qs:
        assert a.shape == (k,), f"all alpha must have shape ({k},); got {a.shape}"
    if alpha_bits <= 0 or alpha_bits > 16:
        raise ValueError(f"alpha_bits must be in [1, 16], got {alpha_bits}")

    if sparse_bitpacked:
        if pair_indices is None:
            pair_indices = list(range(n_pairs))
        if len(pair_indices) != n_pairs:
            raise ValueError("pair_indices length must match alpha rows")
        if any(idx < 0 or idx > 65535 for idx in pair_indices):
            raise ValueError("pair_indices must fit uint16")
        if len(set(pair_indices)) != len(pair_indices):
            raise ValueError("pair_indices must be unique")
        header = SJKL_BLOCK_V2_MAGIC + struct.pack("<HHB", n_pairs, k, alpha_bits)
        indices = np.array(pair_indices, dtype=np.uint16).tobytes()
        mins_fp16 = np.array(alpha_mins, dtype=np.float16).tobytes()
        steps_fp16 = np.array(alpha_steps, dtype=np.float16).tobytes()
        q_matrix = np.stack(
            [a.astype(_uint_dtype_for_bits(alpha_bits), copy=False) for a in alpha_qs],
            axis=0,
        )
        raw = header + indices + mins_fp16 + steps_fp16 + _pack_bitpacked_uints(q_matrix, bits=alpha_bits)
        return brotli.compress(raw, quality=brotli_quality)

    if alpha_bits <= 8:
        a_dtype = np.uint8
    elif alpha_bits <= 16:
        a_dtype = np.uint16
    else:
        raise ValueError(f"alpha_bits {alpha_bits} > 16 not supported")

    header = SJKL_BLOCK_MAGIC + struct.pack("<HHB", n_pairs, k, alpha_bits)
    mins_fp16 = np.array(alpha_mins, dtype=np.float16).tobytes()
    steps_fp16 = np.array(alpha_steps, dtype=np.float16).tobytes()
    qs = np.stack([a.astype(a_dtype) for a in alpha_qs], axis=0).tobytes()
    raw = header + mins_fp16 + steps_fp16 + qs
    return brotli.compress(raw, quality=brotli_quality)


def unpack_alpha_block(payload: bytes) -> tuple[list[np.ndarray], list[float], list[float], int]:
    """Inverse of pack_alpha_block.

    Returns (alpha_qs, alpha_mins, alpha_steps, alpha_bits).
    """
    qs, mins, steps, alpha_bits, _pair_indices = unpack_alpha_block_with_indices(payload)
    return qs, mins, steps, alpha_bits


def unpack_alpha_block_with_indices(
    payload: bytes,
) -> tuple[list[np.ndarray], list[float], list[float], int, list[int] | None]:
    """Inverse of pack_alpha_block, preserving sparse pair indices when present."""
    import brotli

    raw = brotli.decompress(payload)
    if raw[:4] == SJKL_BLOCK_V2_MAGIC:
        n_pairs, k, alpha_bits = struct.unpack("<HHB", raw[4:9])
        if n_pairs <= 0 or k <= 0:
            raise ValueError("empty sparse alpha block")
        cursor = 9
        indices_end = cursor + 2 * n_pairs
        mins_end = indices_end + 2 * n_pairs
        steps_end = mins_end + 2 * n_pairs
        packed_len = math.ceil(n_pairs * k * alpha_bits / 8)
        qs_end = steps_end + packed_len
        if qs_end != len(raw):
            raise ValueError(
                f"sparse alpha block length mismatch: expected {qs_end}, got {len(raw)}"
            )
        pair_indices = np.frombuffer(raw[cursor:indices_end], dtype=np.uint16).astype(np.int64).tolist()
        if len(set(pair_indices)) != len(pair_indices):
            raise ValueError("sparse alpha block contains duplicate pair indices")
        mins = np.frombuffer(raw[indices_end:mins_end], dtype=np.float16).astype(np.float32).copy()
        steps = np.frombuffer(raw[mins_end:steps_end], dtype=np.float16).astype(np.float32).copy()
        q_flat = _unpack_bitpacked_uints(raw[steps_end:qs_end], count=n_pairs * k, bits=alpha_bits)
        qs_arr = q_flat.reshape(n_pairs, k)
        return [qs_arr[i].copy() for i in range(n_pairs)], mins.tolist(), steps.tolist(), int(alpha_bits), pair_indices

    if raw[:4] != SJKL_BLOCK_MAGIC:
        raise ValueError(f"bad block magic: {raw[:4]!r}")
    n_pairs, k, alpha_bits = struct.unpack("<HHB", raw[4:9])
    cursor = 9
    mins = np.frombuffer(raw[cursor : cursor + 2 * n_pairs], dtype=np.float16).astype(np.float32).copy()
    cursor += 2 * n_pairs
    steps = np.frombuffer(raw[cursor : cursor + 2 * n_pairs], dtype=np.float16).astype(np.float32).copy()
    cursor += 2 * n_pairs
    if alpha_bits <= 8:
        a_dtype = np.uint8
        per = 1
    else:
        a_dtype = np.uint16
        per = 2
    qs_flat = np.frombuffer(raw[cursor : cursor + per * n_pairs * k], dtype=a_dtype).copy()
    qs = [qs_flat[i * k : (i + 1) * k] for i in range(n_pairs)]
    return qs, mins.tolist(), steps.tolist(), int(alpha_bits), None


def pack_full_sjkl_payload(basis_bytes: bytes, block_bytes: bytes) -> bytes:
    """Combine (basis, block) into a single sjkl.bin payload with TOC.

    Layout: MAGIC(4='SJKL') | basis_len(uint32) | block_len(uint32) |
            basis_bytes_no_magic | block_bytes
    Note basis_bytes already starts with SJKL magic; we strip it to dedupe.
    """
    if basis_bytes[:4] != SJKL_MAGIC:
        raise ValueError("basis_bytes missing SJKL magic")
    inner_basis = basis_bytes[4:]
    return (
        SJKL_MAGIC
        + struct.pack("<II", len(inner_basis), len(block_bytes))
        + inner_basis
        + block_bytes
    )


def unpack_full_sjkl_payload(payload: bytes) -> tuple[SJKLBasis, list[np.ndarray], list[float], list[float], int]:
    """Inverse of pack_full_sjkl_payload."""
    basis, qs, mins, steps, alpha_bits, _pair_indices = unpack_full_sjkl_payload_with_indices(payload)
    return basis, qs, mins, steps, alpha_bits


def unpack_full_sjkl_payload_with_indices(
    payload: bytes,
) -> tuple[SJKLBasis, list[np.ndarray], list[float], list[float], int, list[int] | None]:
    """Inverse of pack_full_sjkl_payload, preserving sparse pair indices."""
    if payload[:4] != SJKL_MAGIC:
        raise ValueError(f"bad payload magic: {payload[:4]!r}")
    basis_len, block_len = struct.unpack("<II", payload[4:12])
    cursor = 12
    inner_basis = payload[cursor : cursor + basis_len]
    cursor += basis_len
    block_bytes = payload[cursor : cursor + block_len]
    basis = unpack_sjkl_basis(SJKL_MAGIC + inner_basis)
    qs, mins, steps, ab, pair_indices = unpack_alpha_block_with_indices(block_bytes)
    return basis, qs, mins, steps, ab, pair_indices


def _build_alpha_rows(
    *,
    renderer_out: torch.Tensor,
    gt_pairs: torch.Tensor,
    basis_cpu: SJKLBasis,
    target_slot: int,
    alpha_bits: int,
    residual_gain: float,
) -> tuple[list[np.ndarray], list[float], list[float], list[float], list[float]]:
    from tac.sjkl_basis import project_residual

    alpha_qs: list[np.ndarray] = []
    alpha_mins: list[float] = []
    alpha_steps: list[float] = []
    alpha_energy: list[float] = []
    residual_l2: list[float] = []
    scale = basis_cpu.scale.detach().cpu().numpy().astype(np.float32)
    for i in range(int(renderer_out.shape[0])):
        r = residual_for_target_slot(
            renderer_out[i],
            gt_pairs[i],
            target_slot=target_slot,
        )
        alpha = project_residual(r, basis_cpu).detach().cpu().numpy().astype(np.float32)
        alpha_energy.append(float(np.sum((alpha * scale) ** 2)))
        residual_l2.append(float(r.float().pow(2).sum().sqrt().item()))
        a_q, a_min, a_step = _quantize_alpha(alpha * float(residual_gain), alpha_bits=alpha_bits)
        alpha_qs.append(a_q)
        alpha_mins.append(a_min)
        alpha_steps.append(a_step)
    return alpha_qs, alpha_mins, alpha_steps, alpha_energy, residual_l2


def parse_pair_indices_arg(value: str | None) -> list[int] | None:
    """Parse explicit absolute pair indices from CLI text or a small local file."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        raise ValueError("--pair-indices must not be empty")

    maybe_path = Path(text)
    if maybe_path.exists():
        text = maybe_path.read_text().strip()
        if not text:
            raise ValueError(f"pair index file is empty: {maybe_path}")
        if text.startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("pair index JSON must be a list")
            indices = payload
        else:
            indices = [part for line in text.splitlines() for part in line.replace(",", " ").split()]
    else:
        indices = [part for part in text.replace(",", " ").split()]

    out: list[int] = []
    for item in indices:
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise ValueError(f"pair index must be an integer, got {item!r}")
        try:
            idx = int(item)
        except ValueError as exc:
            raise ValueError(f"pair index must be an integer, got {item!r}") from exc
        out.append(idx)
    if not out:
        raise ValueError("--pair-indices must contain at least one index")
    if any(idx < 0 for idx in out):
        raise ValueError("pair indices must be non-negative")
    if len(set(out)) != len(out):
        raise ValueError("pair indices must be unique")
    return out


def _select_pair_rows(
    *,
    pair_selection: str,
    max_encoded_pairs: int | None,
    n_pairs: int,
    alpha_energy: list[float],
    residual_l2: list[float],
    explicit_pair_indices: list[int] | None = None,
    source_pair_indices: list[int] | None = None,
) -> list[int]:
    if source_pair_indices is None:
        source_pair_indices = list(range(n_pairs))
    if len(source_pair_indices) != n_pairs:
        raise ValueError("source_pair_indices length must match n_pairs")
    if explicit_pair_indices is not None:
        if max_encoded_pairs is not None:
            raise ValueError("--pair-indices and --max-encoded-pairs are mutually exclusive")
        if pair_selection not in ("alpha_energy", "first", "residual_l2", "explicit"):
            raise ValueError(f"unknown pair_selection={pair_selection!r}")
        row_by_pair = {int(pair_idx): row for row, pair_idx in enumerate(source_pair_indices)}
        missing = [int(pair_idx) for pair_idx in explicit_pair_indices if int(pair_idx) not in row_by_pair]
        if missing:
            raise ValueError(f"explicit pair indices not present in source rows: {missing[:10]}")
        return [row_by_pair[int(pair_idx)] for pair_idx in explicit_pair_indices]
    if max_encoded_pairs is None:
        return list(range(n_pairs))
    if max_encoded_pairs <= 0:
        raise ValueError(f"max_encoded_pairs must be positive, got {max_encoded_pairs}")
    count = min(int(max_encoded_pairs), n_pairs)
    if pair_selection == "first":
        return list(range(count))
    if pair_selection == "alpha_energy":
        scores = alpha_energy
    elif pair_selection == "residual_l2":
        scores = residual_l2
    else:
        raise ValueError(f"unknown pair_selection={pair_selection!r}")
    return sorted(sorted(range(n_pairs), key=lambda idx: (-scores[idx], idx))[:count])


def repack_sjkl_payload(
    *,
    source_sjkl_bin: Path,
    out: Path,
    manifest: Path | None = None,
    alpha_bits: int | None = None,
    residual_gain: float = 1.0,
    max_encoded_pairs: int | None = None,
    pair_selection: str = "alpha_energy",
    explicit_pair_indices: list[int] | None = None,
    repack_k: int | None = None,
    repack_basis_grid_h: int | None = None,
    repack_basis_grid_w: int | None = None,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    """Create a smaller deterministic SJ-KL payload from existing CUDA-built bytes."""
    source_payload = source_sjkl_bin.read_bytes()
    basis, qs, mins, steps, source_alpha_bits, pair_indices = unpack_full_sjkl_payload_with_indices(source_payload)
    source_pair_indices = pair_indices or list(range(len(qs)))
    alpha_bits = int(source_alpha_bits if alpha_bits is None else alpha_bits)
    if residual_gain < 0:
        raise ValueError(f"residual_gain must be non-negative, got {residual_gain}")

    q_matrix = np.stack(qs, axis=0)
    alpha_matrix = np.stack(
        [
            np.asarray(mins[row], dtype=np.float32)
            + np.asarray(steps[row], dtype=np.float32) * q_matrix[row].astype(np.float32)
            for row in range(q_matrix.shape[0])
        ],
        axis=0,
    )

    if repack_k is not None:
        if repack_k <= 0 or repack_k > alpha_matrix.shape[1]:
            raise ValueError(f"repack_k must be in [1, {alpha_matrix.shape[1]}], got {repack_k}")
        basis = SJKLBasis(
            basis_coarse=basis.basis_coarse[:repack_k].clone(),
            scale=basis.scale[:repack_k].clone(),
            target_h=basis.target_h,
            target_w=basis.target_w,
        )
        alpha_matrix = alpha_matrix[:, :repack_k]

    if repack_basis_grid_h is not None or repack_basis_grid_w is not None:
        grid_h = int(repack_basis_grid_h or basis.basis_coarse.shape[2])
        grid_w = int(repack_basis_grid_w or basis.basis_coarse.shape[3])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"basis grid must be positive, got {grid_h}x{grid_w}")
        basis = SJKLBasis(
            basis_coarse=F.interpolate(
                basis.basis_coarse,
                size=(grid_h, grid_w),
                mode="bilinear",
                align_corners=False,
            ),
            scale=basis.scale.clone(),
            target_h=basis.target_h,
            target_w=basis.target_w,
        ).renormalize()

    scale = basis.scale.detach().cpu().numpy().astype(np.float32)
    alpha_energy = np.sum((alpha_matrix[:, : len(scale)] * scale.reshape(1, -1)) ** 2, axis=1)
    selected_rows = _select_pair_rows(
        pair_selection=pair_selection,
        max_encoded_pairs=max_encoded_pairs,
        n_pairs=alpha_matrix.shape[0],
        alpha_energy=alpha_energy.astype(np.float64).tolist(),
        residual_l2=alpha_energy.astype(np.float64).tolist(),
        explicit_pair_indices=explicit_pair_indices,
        source_pair_indices=[int(x) for x in source_pair_indices],
    )

    out_qs: list[np.ndarray] = []
    out_mins: list[float] = []
    out_steps: list[float] = []
    out_pair_indices: list[int] = []
    for row in selected_rows:
        a_q, a_min, a_step = _quantize_alpha(
            alpha_matrix[row, : len(scale)] * float(residual_gain),
            alpha_bits=alpha_bits,
        )
        out_qs.append(a_q)
        out_mins.append(a_min)
        out_steps.append(a_step)
        out_pair_indices.append(int(source_pair_indices[row]))

    basis_bytes = pack_sjkl_basis(basis, brotli_quality=brotli_quality)
    block_bytes = pack_alpha_block(
        out_qs,
        out_mins,
        out_steps,
        alpha_bits=alpha_bits,
        brotli_quality=brotli_quality,
        pair_indices=out_pair_indices,
        sparse_bitpacked=True,
    )
    payload = pack_full_sjkl_payload(basis_bytes, block_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    result = {
        "schema": "sjkl_repack_manifest_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "deterministic_repack_no_score",
        "source_sjkl": {
            "path": str(source_sjkl_bin),
            "bytes": int(source_sjkl_bin.stat().st_size),
            "sha256": _sha256_file(source_sjkl_bin),
            "source_alpha_bits": int(source_alpha_bits),
            "source_rows": int(len(qs)),
            "source_k": int(q_matrix.shape[1]),
            "source_pair_indices": "implicit_sequential" if pair_indices is None else "explicit_sparse",
        },
        "out": {
            "path": str(out),
            "bytes": int(out.stat().st_size),
            "sha256": _sha256_file(out),
        },
        "out_bytes": int(out.stat().st_size),
        "total_bytes": int(out.stat().st_size),
        "total_sha256": _sha256_file(out),
        "basis_bytes": len(basis_bytes),
        "coefficient_block_bytes": len(block_bytes),
        "k": int(basis.basis_coarse.shape[0]),
        "basis_grid_h": int(basis.basis_coarse.shape[2]),
        "basis_grid_w": int(basis.basis_coarse.shape[3]),
        "target_h": int(basis.target_h),
        "target_w": int(basis.target_w),
        "alpha_bits": int(alpha_bits),
        "alpha_block_format": "sparse_bitpacked_v2",
        "residual_gain": float(residual_gain),
        "pair_selection": "explicit" if explicit_pair_indices is not None else pair_selection,
        "requested_pair_indices": None
        if explicit_pair_indices is None
        else [int(x) for x in explicit_pair_indices],
        "selected_pair_count": int(len(out_pair_indices)),
        "selected_pair_indices": out_pair_indices,
        "max_encoded_pairs": None if max_encoded_pairs is None else int(max_encoded_pairs),
        "rate_contribution": 25.0 * int(out.stat().st_size) / 37545489,
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
    }
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def residual_for_target_slot(
    renderer_frame: torch.Tensor,
    gt_pair: torch.Tensor,
    *,
    target_slot: int,
) -> torch.Tensor:
    """Return the residual for the exact JointFrameGenerator pair slot."""
    if target_slot not in (0, 1):
        raise ValueError(f"target_slot must be 0 or 1, got {target_slot}")
    if renderer_frame.dim() != 3:
        raise ValueError(f"renderer_frame must be (3,H,W), got {tuple(renderer_frame.shape)}")
    if gt_pair.dim() != 4 or gt_pair.shape[0] != 2:
        raise ValueError(f"gt_pair must be (2,3,H,W), got {tuple(gt_pair.shape)}")
    if tuple(gt_pair[target_slot].shape) != tuple(renderer_frame.shape):
        raise ValueError(
            "renderer/GT target slot shape mismatch: "
            f"{tuple(renderer_frame.shape)} vs {tuple(gt_pair[target_slot].shape)}"
        )
    return (gt_pair[target_slot].float() - renderer_frame.float()).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renderer-output", type=Path,
                    help="Path to .pt file containing renderer outputs as a "
                         "tensor of shape (n_pairs, 3, H, W) — frame1 outputs "
                         "(or frame2 if --base frame2 specified).")
    ap.add_argument("--gt-pairs", type=Path,
                    help="Path to .pt file with GT frames as a tensor of "
                         "shape (n_pairs, 2, 3, H, W) used for scorer eval "
                         "and residual computation.")
    ap.add_argument("--source-sjkl-bin", type=Path,
                    help="Repack an existing SJ-KL payload without loading scorers.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output sjkl.bin path.")
    ap.add_argument("--k", type=int, default=8, help="Top-k eigenvectors.")
    ap.add_argument("--basis-grid-h", type=int, default=32)
    ap.add_argument("--basis-grid-w", type=int, default=24)
    ap.add_argument(
        "--alpha-bits",
        type=int,
        default=None,
        help="Alpha quantization bits. Defaults to 6 for tensor builds and to the source payload bits for --source-sjkl-bin repacks.",
    )
    ap.add_argument("--alpha-block-format", choices=("legacy", "sparse"), default="legacy")
    ap.add_argument("--max-encoded-pairs", type=int, default=None)
    ap.add_argument(
        "--pair-selection",
        choices=("first", "alpha_energy", "residual_l2", "explicit"),
        default="alpha_energy",
    )
    ap.add_argument(
        "--pair-indices",
        type=str,
        default=None,
        help=(
            "Comma/space separated absolute pair indices, or a small file "
            "containing a JSON list or newline/comma separated integers. "
            "Overrides score-proxy row selection and remains non-scoring "
            "until the packed archive receives exact CUDA auth eval."
        ),
    )
    ap.add_argument("--residual-gain", type=float, default=1.0)
    ap.add_argument("--repack-k", type=int, default=None)
    ap.add_argument("--repack-basis-grid-h", type=int, default=None)
    ap.add_argument("--repack-basis-grid-w", type=int, default=None)
    ap.add_argument("--target-slot", type=int, default=0,
                    help="GT pair slot corrected by runtime. robust_current "
                         "currently supports only slot 0 (fake1).")
    ap.add_argument("--n-anchor-pairs", type=int, default=30,
                    help="Number of frame pairs to use for Fisher averaging.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda/mps/cpu. CUDA STRONGLY PREFERRED for score validity.")
    ap.add_argument("--allow-non-cuda", action="store_true",
                    help="Allow advisory CPU/MPS build. Production dispatches "
                         "must leave this unset so CUDA failures fail closed.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional: write a JSON manifest with byte counts + provenance.")
    args = ap.parse_args()
    explicit_pair_indices = parse_pair_indices_arg(args.pair_indices)
    if args.source_sjkl_bin is not None:
        result = repack_sjkl_payload(
            source_sjkl_bin=args.source_sjkl_bin,
            out=args.out,
            manifest=args.manifest,
            alpha_bits=args.alpha_bits,
            residual_gain=args.residual_gain,
            max_encoded_pairs=args.max_encoded_pairs,
            pair_selection=args.pair_selection,
            explicit_pair_indices=explicit_pair_indices,
            repack_k=args.repack_k,
            repack_basis_grid_h=args.repack_basis_grid_h,
            repack_basis_grid_w=args.repack_basis_grid_w,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.renderer_output is None or args.gt_pairs is None:
        raise ValueError("--renderer-output and --gt-pairs are required unless --source-sjkl-bin is used")
    if args.alpha_bits is None:
        args.alpha_bits = 6
    if args.target_slot != 0:
        raise ValueError(
            "--target-slot must be 0 for the current robust_current SJ-KL "
            "runtime, which applies residuals to JointFrameGenerator fake1."
        )

    device = resolve_sjkl_build_device(args.device, allow_non_cuda=args.allow_non_cuda)

    print(f"[sjkl-basis] loading renderer output from {args.renderer_output}")
    renderer_out = torch.load(args.renderer_output, map_location="cpu", weights_only=True)
    print(f"[sjkl-basis] loading GT pairs from {args.gt_pairs}")
    gt_pairs = torch.load(args.gt_pairs, map_location="cpu", weights_only=True)

    assert renderer_out.dim() == 4, f"renderer_output must be (n,3,H,W); got {renderer_out.shape}"
    assert gt_pairs.dim() == 5, f"gt_pairs must be (n,2,3,H,W); got {gt_pairs.shape}"
    n_pairs, C, H, W = renderer_out.shape
    assert gt_pairs.shape[0] == n_pairs, "n_pairs mismatch"

    print(f"[sjkl-basis] loading scorers on {device}")
    segnet, posenet = _load_scorers(device)

    # Pick anchor frames evenly across the sequence
    anchor_idx = np.linspace(0, n_pairs - 1, args.n_anchor_pairs, dtype=np.int64)
    anchor_pairs = gt_pairs[anchor_idx].to(device=device, dtype=torch.float32)

    print(f"[sjkl-basis] computing SJ-KL basis: k={args.k}, n_anchors={args.n_anchor_pairs}, "
          f"grid={args.basis_grid_h}x{args.basis_grid_w}")
    basis = compute_sjkl_basis(
        segnet, posenet, anchor_pairs,
        k=args.k,
        basis_grid_h=args.basis_grid_h,
        basis_grid_w=args.basis_grid_w,
        seed=args.seed,
    )
    basis_bytes = pack_sjkl_basis(basis)
    print(f"[sjkl-basis] basis packed: {len(basis_bytes)} bytes")

    # Move basis to CPU for residual loop (no scorer needed past this point)
    basis_cpu = SJKLBasis(
        basis_coarse=basis.basis_coarse.cpu(),
        scale=basis.scale.cpu(),
        target_h=basis.target_h,
        target_w=basis.target_w,
    )
    alpha_qs, alpha_mins, alpha_steps, alpha_energy, residual_l2 = _build_alpha_rows(
        renderer_out=renderer_out,
        gt_pairs=gt_pairs,
        basis_cpu=basis_cpu,
        target_slot=args.target_slot,
        alpha_bits=args.alpha_bits,
        residual_gain=args.residual_gain,
    )
    selected_rows = _select_pair_rows(
        pair_selection=args.pair_selection,
        max_encoded_pairs=args.max_encoded_pairs,
        n_pairs=n_pairs,
        alpha_energy=alpha_energy,
        residual_l2=residual_l2,
        explicit_pair_indices=explicit_pair_indices,
        source_pair_indices=list(range(n_pairs)),
    )
    sparse = (
        args.alpha_block_format == "sparse"
        or args.max_encoded_pairs is not None
        or explicit_pair_indices is not None
    )
    selected_qs = [alpha_qs[i] for i in selected_rows]
    selected_mins = [alpha_mins[i] for i in selected_rows]
    selected_steps = [alpha_steps[i] for i in selected_rows]

    block_bytes = pack_alpha_block(
        selected_qs,
        selected_mins,
        selected_steps,
        alpha_bits=args.alpha_bits,
        pair_indices=selected_rows,
        sparse_bitpacked=sparse,
    )
    print(f"[sjkl-basis] alpha block packed: {len(block_bytes)} bytes")

    payload = pack_full_sjkl_payload(basis_bytes, block_bytes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(payload)
    print(f"[sjkl-basis] wrote {args.out} ({len(payload)} bytes total)")

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({
            "schema": "sjkl_residual_manifest_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "cuda_build_no_score"
            if device.type == "cuda"
            else "advisory_non_cuda_build",
            "out": {
                "path": str(args.out),
                "bytes": int(args.out.stat().st_size),
                "sha256": _sha256_file(args.out),
            },
            "out_bytes": int(args.out.stat().st_size),
            "out_sha256": _sha256_file(args.out),
            "inputs": {
                "renderer_output": {
                    "path": str(args.renderer_output),
                    "bytes": int(args.renderer_output.stat().st_size),
                    "sha256": _sha256_file(args.renderer_output),
                    "shape": list(renderer_out.shape),
                    "dtype": str(renderer_out.dtype),
                },
                "gt_pairs": {
                    "path": str(args.gt_pairs),
                    "bytes": int(args.gt_pairs.stat().st_size),
                    "sha256": _sha256_file(args.gt_pairs),
                    "shape": list(gt_pairs.shape),
                    "dtype": str(gt_pairs.dtype),
                },
            },
            "n_pairs": int(n_pairs),
            "k": int(args.k),
            "basis_grid_h": int(args.basis_grid_h),
            "basis_grid_w": int(args.basis_grid_w),
            "alpha_bits": int(args.alpha_bits),
            "alpha_block_format": "sparse_bitpacked_v2" if sparse else "legacy_v1",
            "residual_gain": float(args.residual_gain),
            "pair_selection": "explicit" if explicit_pair_indices is not None else args.pair_selection,
            "requested_pair_indices": explicit_pair_indices,
            "selected_pair_count": int(len(selected_rows)),
            "selected_pair_indices": [int(x) for x in selected_rows],
            "max_encoded_pairs": args.max_encoded_pairs,
            "target_slot": int(args.target_slot),
            "runtime_target": "robust_current JointFrameGenerator pair slot 0 / fake1",
            "n_anchor_pairs": int(args.n_anchor_pairs),
            "anchor_indices": [int(x) for x in anchor_idx.tolist()],
            "seed": int(args.seed),
            "requested_device": str(args.device),
            "actual_device": str(device),
            "device": str(device),
            "allow_non_cuda": bool(args.allow_non_cuda),
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "basis_bytes": len(basis_bytes),
            "basis_sha256": _sha256_bytes(basis_bytes),
            "coefficient_block_bytes": len(block_bytes),
            "coefficient_block_sha256": _sha256_bytes(block_bytes),
            "block_bytes": len(block_bytes),
            "total_bytes": len(payload),
            "total_sha256": _sha256_bytes(payload),
            "rate_contribution": 25.0 * len(payload) / 37545489,  # contest formula
        }, indent=2, sort_keys=True))
        print(f"[sjkl-basis] wrote manifest {args.manifest}")


def apply_sjkl_at_decode(
    fake1: torch.Tensor,
    fake2: torch.Tensor,
    sjkl_payload: bytes,
    pair_idx: int,
    *,
    base: str = "frame1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-side helper that mirrors pr67_inflate.py:878-884 actuator pattern.

    Returns updated (fake1, fake2). If base == "frame1" (default), adds the
    SJ-KL residual to fake1; if "frame2", adds to fake1 derived from fake2.
    """
    basis, qs, mins, steps, _ab, pair_indices = unpack_full_sjkl_payload_with_indices(sjkl_payload)
    if pair_indices is None:
        row = int(pair_idx)
    else:
        row_by_pair = {int(idx): pos for pos, idx in enumerate(pair_indices)}
        if int(pair_idx) not in row_by_pair:
            return fake1, fake2
        row = row_by_pair[int(pair_idx)]
    if row >= len(qs):
        # No SJ-KL for this pair (e.g., partial coverage); pass through.
        return fake1, fake2
    delta = decode_residual_lazy(qs[row], mins[row], steps[row], basis)
    delta = delta.to(device=fake1.device, dtype=fake1.dtype)
    if base == "frame2":
        fake1_new = fake2 + delta
    else:
        fake1_new = fake1 + delta
    return fake1_new, fake2


def decode_residual_lazy(
    alpha_q: np.ndarray,
    alpha_min: float,
    alpha_step: float,
    basis: SJKLBasis,
) -> torch.Tensor:
    """Convenience wrapper around tac.sjkl_basis.decode_residual."""
    from tac.sjkl_basis import decode_residual
    return decode_residual(alpha_q, alpha_min, alpha_step, basis)


if __name__ == "__main__":
    main()
