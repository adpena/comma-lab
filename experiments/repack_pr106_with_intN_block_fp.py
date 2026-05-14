#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Repack PR106 HNeRV decoder via signed intN block-FP (parameterized 4..8).

Generalization of experiments/repack_pr106_with_int4_block_fp.py to take
arbitrary bit width via --bits N. Uses the canonical int-N codec from
experiments/block_fp_intN_codec_sketch.py.

Pareto sweet spots from the int3..int8 sweep are forensic byte previews only.
Exact int4 T4 eval later showed catastrophic scorer-basin drift, so no intN
candidate is dispatch-ready without a SHA-tied distortion model,
scorer-basin parity report, or exact positive CUDA evidence.

Each variant produces an apogee_intN-format archive with magic byte
0xA0 | bits (so 0xA5 = int5, 0xA6 = int6, etc.). The runtime inflate
adapter must dispatch on the high-nibble + bit-width.

Usage (int5 sweet-spot):
    .venv/bin/python experiments/repack_pr106_with_intN_block_fp.py \\
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \\
        --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\
        --out-dir experiments/results/apogee_int5_repack_20260504_claude/ \\
        --bits 5
"""
from __future__ import annotations

import argparse
import hashlib
import io
import struct
import sys
import zipfile
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import torch

PR106_SRC_PATH = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from block_fp_intN_codec_sketch import encode_intN_block_fp  # type: ignore[import-not-found]
from codec import encode_decoder, quantize_state_dict  # type: ignore[import-not-found]
from tac.repo_io import json_text

CODEC_ID_BROTLI_INT8 = 0
CODEC_ID_INTN_BLOCKFP_BROTLI = 5  # 0x05 distinct from OWV2's 0x01 + apogee_int4's 0x02
APOGEE_INTN_DISPATCH_BLOCKERS = [
    "missing_contest_faithful_distortion_model",
    "missing_scorer_basin_parity_gate",
    "byte_only_prediction_not_score_evidence",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _magic_for_bits(bits: int) -> int:
    """0xA0 high-nibble + bits low-nibble. int4=0xA4, int5=0xA5, ..., int8=0xA8."""
    if not 4 <= bits <= 8:
        raise ValueError(f"bits must be in [4, 8], got {bits}")
    return 0xA0 | bits


def _is_intn_eligible(name: str, t: torch.Tensor) -> bool:
    """4-D Conv2d weight only (matching OWV2 + int4 eligibility)."""
    return t.dim() == 4 and ".weight" in name and t.shape[0] >= 1


def _encode_brotli_int8_single(name: str, t: torch.Tensor) -> bytes:
    sd_one = {name: t}
    q_sd = quantize_state_dict(sd_one)
    return encode_decoder(q_sd)


def _encode_intn_blockfp_with_brotli(t: torch.Tensor, *, bits: int, block_size: int = 128) -> bytes:
    """intN block-FP encode then brotli compress."""
    blob = encode_intN_block_fp(t.to(torch.float32), block_size=block_size, bits=bits)
    return brotli.compress(blob, quality=11)


def _build_apogee_intn_blob(
    encoded: dict[str, tuple[int, bytes]],
    latent_brotli: bytes,
    *,
    bits: int,
) -> bytes:
    """apogee_intN 0.bin layout (extension of apogee_int4 with bits-encoded magic):

        magic(1B) = 0xA0 | bits  (int5 = 0xA5, int6 = 0xA6, ...)
        n_codecs(1B)
        for each entry:
            codec_id(1B), name_len(1B), name_utf8 bytes,
            shape_ndim(1B), payload_len(4B uint32 LE), payload bytes
        latent_len(4B uint32 LE), latent_brotli bytes

    The inflate adapter dispatches on the magic byte's low nibble to know
    which bits value to pass to decode_intN_block_fp.
    """
    buf = io.BytesIO()
    buf.write(bytes([_magic_for_bits(bits)]))
    buf.write(bytes([len(encoded) & 0xFF]))
    for name, (codec_id, payload) in encoded.items():
        buf.write(bytes([codec_id & 0xFF]))
        nb = name.encode("utf-8")
        buf.write(bytes([len(nb) & 0xFF]))
        buf.write(nb)
        buf.write(bytes([0]))  # shape_ndim placeholder
        buf.write(struct.pack("<I", len(payload)))
        buf.write(payload)
    buf.write(struct.pack("<I", len(latent_brotli)))
    buf.write(latent_brotli)
    return buf.getvalue()


def _parse_pr106_packed_for_latents(bin_bytes: bytes) -> tuple[bytes, dict]:
    dec_len = int.from_bytes(bin_bytes[1:4], "little")
    latent_bytes = bin_bytes[4 + dec_len:]
    return latent_bytes, {"dec_len": dec_len}


# Per-bits empirical preview (from block_fp_intN_codec_sketch.py sweep against PR106)
_BITS_PROFILE = {
    4: {"rel_err_pct": 7.09, "risk": "HIGH",   "predicted_band": "[0.155, 0.180]"},
    5: {"rel_err_pct": 3.31, "risk": "MEDIUM", "predicted_band": "[0.180, 0.196]"},
    6: {"rel_err_pct": 1.55, "risk": "LOW",    "predicted_band": "[0.190, 0.204]"},
    7: {"rel_err_pct": 0.79, "risk": "VERY LOW", "predicted_band": "[0.198, 0.208]"},
    8: {"rel_err_pct": 0.24, "risk": "ALMOST LOSSLESS", "predicted_band": "[0.196, 0.207]"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--pr106-archive", type=Path, required=True)
    parser.add_argument("--bits", type=int, required=True, choices=[4, 5, 6, 7, 8],
                        help="Bits per quantized weight (4..8). int5 is the sweet spot.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_dict = torch.load(args.state_dict, map_location="cpu", weights_only=False)
    profile = _BITS_PROFILE[args.bits]
    print(f"[repack-int{args.bits}] state_dict: {len(state_dict)} tensors")
    print(f"[repack-int{args.bits}] forensic byte profile: {profile['risk']} risk, "
          f"~{profile['rel_err_pct']:.2f}% rel err per weight, "
          f"historical predicted score band {profile['predicted_band']}")

    encoded: dict[str, tuple[int, bytes]] = {}
    for name, t in state_dict.items():
        if _is_intn_eligible(name, t):
            payload = _encode_intn_blockfp_with_brotli(t, bits=args.bits, block_size=args.block_size)
            encoded[name] = (CODEC_ID_INTN_BLOCKFP_BROTLI, payload)
            print(f"  [int{args.bits}] {name}: {len(payload)} bytes")
        else:
            payload = _encode_brotli_int8_single(name, t)
            encoded[name] = (CODEC_ID_BROTLI_INT8, payload)
            print(f"  [brotli-int8] {name}: {len(payload)} bytes")

    with zipfile.ZipFile(args.pr106_archive) as z:
        bin_bytes = z.read("0.bin")
    latent_brotli, _ = _parse_pr106_packed_for_latents(bin_bytes)
    print(f"[repack-int{args.bits}] harvested latent_brotli: {len(latent_brotli)} bytes")

    new_bin = _build_apogee_intn_blob(encoded, latent_brotli, bits=args.bits)
    print(f"[repack-int{args.bits}] apogee_int{args.bits} 0.bin: {len(new_bin)} bytes "
          f"(magic 0x{_magic_for_bits(args.bits):02X})")

    archive_path = args.out_dir / f"apogee_int{args.bits}_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    pr106_size = args.pr106_archive.stat().st_size
    delta = archive_size - pr106_size
    score_delta = 25.0 * delta / 37545489.0
    print(f"[repack-int{args.bits}] wrote {archive_path}: {archive_size} bytes")
    print(f"[repack-int{args.bits}] PR106: {pr106_size} bytes; delta: {delta:+d} ({100*delta/pr106_size:+.2f}%)")
    print(f"[repack-int{args.bits}] estimated rate-component score Δ vs PR106: {score_delta:+.6f}")
    print(f"[repack-int{args.bits}] forensic only: distortion model is missing; "
          f"~{profile['rel_err_pct']:.2f}% rel err per weight; "
          f"historical predicted band {profile['predicted_band']} is noncanonical")

    metadata = {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "candidate_archive_sha256": _sha256_file(archive_path),
        "source_archive_sha256": _sha256_file(args.pr106_archive),
        "pr106_size_bytes": pr106_size,
        "delta_bytes": delta,
        "rate_component_score_delta_vs_pr106": score_delta,
        "bits": int(args.bits),
        "block_size": int(args.block_size),
        "magic_byte": f"0x{_magic_for_bits(args.bits):02X}",
        "n_intn_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_INTN_BLOCKFP_BROTLI),
        "n_brotli_int8_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_BROTLI_INT8),
        "rel_err_pct_per_weight": profile["rel_err_pct"],
        "distortion_risk": profile["risk"],
        "predicted_score_band": profile["predicted_band"],
        "prediction_status": "forensic_byte_only_invalidated_by_int4_exact_negative",
        "distortion_model_status": "missing",
        "scorer_basin_parity_status": "missing",
        "score_affecting_payload_changed": True,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": APOGEE_INTN_DISPATCH_BLOCKERS,
        "tag": "[design-validation]",
        "council_status": "NOT_REVIEWED",
        "score_claim": False,
        "next_step": (
            "Do not dispatch as a score lane. Build a SHA-tied distortion model "
            "or scorer-basin parity report first, or attach exact positive CUDA "
            "evidence for the exact candidate archive."
        ),
    }
    metadata_path = args.out_dir / "repack_metadata.json"
    metadata_path.write_text(json_text(metadata), encoding="utf-8")
    print(f"[repack-int{args.bits}] wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
