#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/pack_semantic_pose.py
#   source_sha256: 151413b0696a23e26fbc246dd2cbf7f42c3d64f67290e7c61f21ccc1d0a9bfbc
#   source_sha256_scope: exact source_path bytes at source_repo_head before adaptations
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: accounting header only; mx2 behavior changes live in sibling wrappers.
#   adaptations: none
#   score_claim: false
#
"""Pack and verify the semantic renderer plus pose carrier payload."""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np
import torch

from learned_pose_carrier_oracle import quantize_basis
from semantic_renderer_oracle import SemanticTokenRenderer



# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

def pack_signed_int4(codes: torch.Tensor) -> bytes:
    values = codes.detach().cpu().numpy().astype(np.int8, copy=False).reshape(-1)
    nibbles = (values.astype(np.int16) & 0xF).astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.pad(nibbles, (0, 1))
    packed = nibbles[0::2] | (nibbles[1::2] << 4)
    return packed.tobytes()


def unpack_signed_int4(blob: memoryview, count: int) -> tuple[torch.Tensor, memoryview]:
    byte_count = (count + 1) // 2
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    values = np.empty(byte_count * 2, dtype=np.int8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    values[values >= 8] -= 16
    return torch.from_numpy(values[:count].copy()), blob[byte_count:]


def pack_signed_bits(codes: torch.Tensor, bits: int) -> bytes:
    if not 2 <= bits <= 8:
        raise ValueError("signed bit packer supports 2 through 8 bits")
    values = codes.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
    unsigned = (values & ((1 << bits) - 1)).astype(np.uint16, copy=False)
    shifts = np.arange(bits, dtype=np.uint16)
    bitstream = ((unsigned[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitstream, bitorder="little").tobytes()


def unpack_signed_bits(
    blob: memoryview, count: int, bits: int
) -> tuple[torch.Tensor, memoryview]:
    if not 2 <= bits <= 8:
        raise ValueError("signed bit unpacker supports 2 through 8 bits")
    byte_count = (count * bits + 7) // 8
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder="little")[:count * bits]
    bitstream = bitstream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return torch.from_numpy(values.astype(np.int8, copy=False)), blob[byte_count:]


def quantize_coeff(coeff: torch.Tensor, bits: int):
    max_code = (1 << (bits - 1)) - 1
    scales = coeff.abs().amax(dim=0).clamp_min(1e-8) / max_code
    dtype = torch.int8 if bits <= 8 else torch.int16
    codes = (coeff / scales).round().clamp(-max_code, max_code).to(dtype)
    return codes.float() * scales, codes, scales


def pack_signed_int12(codes: torch.Tensor) -> bytes:
    values = codes.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
    unsigned = (values.astype(np.int32) & 0xFFF).astype(np.uint16)
    if unsigned.size % 2:
        unsigned = np.pad(unsigned, (0, 1))
    first = unsigned[0::2]
    second = unsigned[1::2]
    packed = np.empty(first.size * 3, dtype=np.uint8)
    packed[0::3] = first & 0xFF
    packed[1::3] = ((first >> 8) & 0xF) | ((second & 0xF) << 4)
    packed[2::3] = second >> 4
    return packed.tobytes()


def unpack_signed_int12(
    blob: memoryview, count: int
) -> tuple[torch.Tensor, memoryview]:
    byte_count = ((count + 1) // 2) * 3
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8).astype(np.uint16)
    values = np.empty((byte_count // 3) * 2, dtype=np.int16)
    values[0::2] = packed[0::3] | ((packed[1::3] & 0xF) << 8)
    values[1::2] = (packed[1::3] >> 4) | (packed[2::3] << 4)
    values[values >= 2048] -= 4096
    return torch.from_numpy(values[:count].copy()), blob[byte_count:]


def quantize_semantic(value: torch.Tensor, bits: int, embedding: bool):
    if bits != 4:
        raise ValueError("binary semantic packer currently requires int4")
    source = value.detach().cpu().float()
    if source.ndim < 2:
        stored = source.to(torch.float16)
        return stored.float(), stored.numpy().tobytes()
    limit = 7
    reduce_dims = tuple(range(source.ndim - 1)) if embedding else tuple(range(1, source.ndim))
    scale = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
    scale = scale.to(torch.float16).clamp(min=_FP16_MIN_POSITIVE)
    q = (source / scale.float()).round().clamp(-limit, limit).to(torch.int8)
    restored = q.float() * scale.float()
    return restored, scale.reshape(-1).numpy().tobytes() + pack_signed_int4(q)


def pack_semantic(checkpoint):
    bits = int(checkpoint["quant_bits"])
    packed = bytearray()
    expected = {}
    for name, value in checkpoint["state_dict"].items():
        restored, blob = quantize_semantic(
            value, bits, embedding=name.endswith("embed.weight")
        )
        expected[name] = restored
        packed.extend(blob)
    return bytes(packed), expected


def unpack_semantic(blob: bytes, config, template_state):
    remaining = memoryview(blob)
    restored = {}
    for name, template in template_state.items():
        shape = tuple(template.shape)
        count = template.numel()
        if template.ndim < 2:
            byte_count = count * 2
            value = np.frombuffer(remaining[:byte_count], dtype=np.float16).copy()
            restored[name] = torch.from_numpy(value).reshape(shape).float()
            remaining = remaining[byte_count:]
            continue
        scale_count = shape[-1] if name.endswith("embed.weight") else shape[0]
        scale_bytes = scale_count * 2
        scale = torch.from_numpy(
            np.frombuffer(remaining[:scale_bytes], dtype=np.float16).copy()
        ).float()
        remaining = remaining[scale_bytes:]
        q, remaining = unpack_signed_int4(remaining, count)
        if name.endswith("embed.weight"):
            scale_shape = [1] * len(shape)
            scale_shape[-1] = scale_count
        else:
            scale_shape = [1] * len(shape)
            scale_shape[0] = scale_count
        restored[name] = q.reshape(shape).float() * scale.reshape(scale_shape)
    if remaining:
        raise ValueError("semantic payload has trailing bytes")
    return restored


def pack_carrier(checkpoint, basis_bits, coeff_bits):
    basis = checkpoint["basis"].float()
    coeff = checkpoint["coeff"].float()
    basis_q, basis_codes, basis_scales = quantize_basis(basis, basis_bits)
    coeff_q, coeff_codes, coeff_scales = quantize_coeff(coeff, coeff_bits)
    packed = bytearray()
    packed.extend(basis_scales.detach().cpu().float().numpy().tobytes())
    packed.extend(pack_signed_bits(basis_codes, basis_bits))
    packed.extend(coeff_scales.detach().cpu().float().numpy().tobytes())
    if coeff_bits == 8:
        packed.extend(coeff_codes.detach().cpu().numpy().astype(np.int8, copy=False).tobytes())
    elif coeff_bits == 12:
        packed.extend(pack_signed_int12(coeff_codes))
    else:
        packed.extend(coeff_codes.detach().cpu().numpy().astype("<i2", copy=False).tobytes())
    expected = {
        "basis": basis_q.cpu(), "coeff": coeff_q.cpu(),
        "basis_codes": basis_codes.cpu(), "coeff_codes": coeff_codes.cpu(),
        "basis_scales": basis_scales.cpu(), "coeff_scales": coeff_scales.cpu(),
    }
    return bytes(packed), expected


def unpack_carrier(blob, basis_shape, coeff_shape, basis_bits, coeff_bits):
    remaining = memoryview(blob)
    basis_dim = basis_shape[0]
    scale_bytes = basis_dim * 4
    basis_scales = torch.from_numpy(
        np.frombuffer(remaining[:scale_bytes], dtype=np.float32).copy()
    )
    remaining = remaining[scale_bytes:]
    basis_count = int(np.prod(basis_shape))
    basis_codes, remaining = unpack_signed_bits(
        remaining, basis_count, basis_bits
    )
    coeff_scales = torch.from_numpy(
        np.frombuffer(remaining[:scale_bytes], dtype=np.float32).copy()
    )
    remaining = remaining[scale_bytes:]
    coeff_count = int(np.prod(coeff_shape))
    if coeff_bits == 8:
        coeff_codes = torch.from_numpy(
            np.frombuffer(remaining[:coeff_count], dtype=np.int8).copy()
        )
        remaining = remaining[coeff_count:]
    elif coeff_bits == 12:
        coeff_codes, remaining = unpack_signed_int12(remaining, coeff_count)
    else:
        byte_count = coeff_count * 2
        coeff_codes = torch.from_numpy(
            np.frombuffer(remaining[:byte_count], dtype="<i2").copy()
        )
        remaining = remaining[byte_count:]
    if remaining:
        raise ValueError("carrier payload has trailing bytes")
    basis = basis_codes.reshape(basis_shape).float() * basis_scales[:, None, None, None]
    coeff = coeff_codes.reshape(coeff_shape).float() * coeff_scales[None]
    return {"basis": basis, "coeff": coeff}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--carrier-bits", type=int, default=4)
    parser.add_argument("--coeff-bits", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    semantic_checkpoint = torch.load(args.semantic, map_location="cpu", weights_only=False)
    carrier_checkpoint = torch.load(args.carrier, map_location="cpu", weights_only=False)
    semantic_blob, semantic_expected = pack_semantic(semantic_checkpoint)
    carrier_blob, carrier_expected = pack_carrier(
        carrier_checkpoint, args.carrier_bits, args.coeff_bits
    )
    config = semantic_checkpoint["config"]
    model = SemanticTokenRenderer(
        width=int(config["width"]), blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]), num_pairs=600,
    )
    semantic_restored = unpack_semantic(semantic_blob, config, model.state_dict())
    if any(not torch.equal(semantic_restored[key], value)
           for key, value in semantic_expected.items()):
        raise RuntimeError("semantic binary round trip changed a dequantized tensor")
    carrier_restored = unpack_carrier(
        carrier_blob, tuple(carrier_checkpoint["basis"].shape),
        tuple(carrier_checkpoint["coeff"].shape), args.carrier_bits,
        args.coeff_bits,
    )
    if not torch.equal(carrier_restored["basis"], carrier_expected["basis"]):
        raise RuntimeError("carrier basis binary round trip changed a tensor")
    if not torch.equal(carrier_restored["coeff"], carrier_expected["coeff"]):
        raise RuntimeError("carrier coefficient binary round trip changed a tensor")

    header = np.asarray([len(semantic_blob), len(carrier_blob)], dtype=np.uint32).tobytes()
    raw = header + semantic_blob + carrier_blob
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(lzma.compress(raw, preset=9))
    result = {
        "verified_exact_tensor_round_trip": True,
        "semantic_bytes": len(semantic_blob),
        "carrier_bytes": len(carrier_blob),
        "header_bytes": len(header),
        "coefficient_bits": args.coeff_bits,
        "basis_bits": args.carrier_bits,
        "raw_combined_bytes": len(raw),
        "lzma_combined_bytes": args.out.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
