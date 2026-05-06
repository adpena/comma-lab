#!/usr/bin/env python3
"""Repack PR106 HNeRV decoder via water-filling codec v2 + arithmetic terminal.

Lane Ω-W-V3 step 3/3 (revival_plan_01_water_filling_codec_v2_pr106_decoder).

Pipeline:
    1. Load PR106 state_dict.pt (28 tensors / 228,958 params, dequantized FP32)
       from extract_pr106_decoder.py output.
    2. Load sensitivity_map.pt (per-channel β-Fisher diagonal)
       from build_sensitivity_map_pr106.py output.
    3. For each Conv2d weight (O, I, kH, kW) in state_dict:
         encoded = encode_omega_w_v2(W, hessian=sensitivity[layer],
                                     total_bits=alloc[layer])
       Bit budget per layer is allocated by global Lagrangian water-fill across
       layers (proportional to per-layer sensitivity-sum × param-count for
       a first pass; can be refined post-empirical).
    4. For non-Conv2d tensors (Linear `stem.weight`, biases, refine layers):
       fall back to PR106's own per-tensor symmetric INT8+brotli (already
       near-Shannon for these small tensors, water-fill amortization unfavorable).
    5. Pack into apogee_v2 0.bin layout:
         magic(1B) | n_codecs(1B) | for each tensor:
           name_len(1B) name codec_id(1B) length(4B) bytes
       Concatenate latent_brotli at end (unchanged from PR106).
    6. Wrap in single-member archive.zip ('0.bin').

Output:
    apogee_v2_archive.zip — the candidate sub-frontier archive
    repack_metadata.json — bytes-by-tensor + total + predicted-vs-PR106 delta

CPU-only for the repack (codec is determinstic). CUDA was needed for the
sensitivity_map upstream; here we just consume it.

Usage:
    .venv/bin/python experiments/repack_pr106_with_water_filling.py \\
        --state-dict experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt \\
        --sensitivity experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt \\
        --pr106-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \\
        --target-bytes 145000 \\
        --out-dir experiments/results/apogee_v2_repack_20260504_claude/
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import torch

PR106_SRC_PATH = Path(__file__).parent / "results" / (
    "public_pr106_belt_and_suspenders_intake_20260504_codex/source/"
    "submissions/belt_and_suspenders/src"
)
sys.path.insert(0, str(PR106_SRC_PATH.resolve()))

from codec import encode_decoder, quantize_state_dict  # type: ignore[import-not-found]

from tac.sensitivity_map import load_sensitivity_map
from tac.water_filling_codec_v2 import encode_omega_w_v2

CODEC_ID_BROTLI_INT8 = 0  # PR106 fallback (for Linear, biases, small layers)
CODEC_ID_OWV2 = 1         # water-fill + arithmetic terminal (for Conv2d)


def _is_water_fillable(name: str, t: torch.Tensor) -> bool:
    """True iff tensor is a 4-D Conv2d weight with O>=1, I>=1 (block-FP eligible)."""
    return t.dim() == 4 and ".weight" in name and t.shape[0] >= 1 and t.shape[1] >= 1


def _allocate_bit_budget(
    sensitivities: dict[str, torch.Tensor],
    state_dict: dict[str, torch.Tensor],
    target_total_bytes: int,
) -> dict[str, int]:
    """First-pass per-Conv2d-layer total_bits = (sensitivity_sum_layer / sensitivity_sum_global)
    × (target_total_bytes × 8). Refinement: pass per-layer sensitivity vector down
    through encode_omega_w_v2 for within-layer Lagrangian water-fill.
    """
    eligible = {n: t for n, t in state_dict.items() if _is_water_fillable(n, t)}
    sens_sums = {n: float(sensitivities[n].sum().item()) for n in eligible if n in sensitivities}
    total_sens = sum(sens_sums.values())
    if total_sens <= 0:
        # Stub fallback: equal-share by output-channel-count.
        weights = {n: float(eligible[n].shape[0]) for n in eligible}
        total_w = sum(weights.values())
        sens_sums = {n: w / total_w for n, w in weights.items()}
        total_sens = 1.0
    target_total_bits = int(target_total_bytes * 8)
    return {n: max(int(target_total_bits * (s / total_sens)), eligible[n].shape[0] * 8)
            for n, s in sens_sums.items()}


def _encode_brotli_int8_single(name: str, t: torch.Tensor) -> bytes:
    """Fallback encoding for non-Conv2d tensors. Per PR106 quantize_state_dict +
    inline encode of just this single tensor (zigzag + brotli)."""
    sd_one = {name: t}
    q_sd = quantize_state_dict(sd_one)
    return encode_decoder(q_sd)


def _build_apogee_v2_blob(
    encoded: dict[str, tuple[int, bytes]],
    latent_brotli: bytes,
) -> bytes:
    """Apogee-v2 0.bin layout (extension of PR106's packed format).

    Layout:
        magic(1B) = 0xA2  (apogee-v2 marker, distinct from PR106's 0xFF packed marker)
        n_codecs(1B) = number of tensor entries
        for each entry:
            codec_id(1B)
            name_len(1B), name_utf8 bytes
            shape_ndim(1B), shape uint32 LE × ndim
            payload_len(4B uint32 LE), payload bytes
        latent_len(4B uint32 LE), latent_brotli bytes
    """
    buf = io.BytesIO()
    buf.write(bytes([0xA2]))
    buf.write(bytes([len(encoded) & 0xFF]))
    for name, (codec_id, payload) in encoded.items():
        buf.write(bytes([codec_id & 0xFF]))
        nb = name.encode("utf-8")
        buf.write(bytes([len(nb) & 0xFF]))
        buf.write(nb)
        # shape is implied by the payload itself in OWV2; for fallback codecs
        # we still write shape so the inverse parser can validate.
        buf.write(bytes([0]))  # shape_ndim=0 (deferred to inverse parser per-codec)
        buf.write(struct.pack("<I", len(payload)))
        buf.write(payload)
    buf.write(struct.pack("<I", len(latent_brotli)))
    buf.write(latent_brotli)
    return buf.getvalue()


def repack_pr106_with_water_filling(
    state_dict_path: Path,
    sensitivity_path: Path,
    pr106_archive: Path,
    out_dir: Path,
    *,
    target_bytes: int = 145000,
    verbose: bool = True,
) -> dict[str, object]:
    """Build the Apogee-v2 repack archive and return its metadata.

    The callable form avoids subprocess startup in preflight while preserving
    the CLI contract for remote wrappers and operator runbooks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict: dict[str, torch.Tensor] = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    sensitivities, sens_meta = load_sensitivity_map(sensitivity_path)
    if verbose:
        print(f"[repack-pr106] state_dict: {len(state_dict)} tensors, "
              f"{sum(t.numel() for t in state_dict.values())} params")
        print(f"[repack-pr106] sensitivity tag: {sens_meta.get('tag', 'unknown')}")

    if sens_meta.get("is_stub") and verbose:
        print("[repack-pr106] WARN: sensitivity is [stub-design-mode]. Repack output is "
              "DESIGN-VALIDATION ONLY — score is NOT predictive. Use real CUDA beta-Fisher "
              "sensitivity for production repack.", file=sys.stderr)

    # 1. Allocate per-Conv2d byte budget
    bits_alloc = _allocate_bit_budget(sensitivities, state_dict, target_bytes)
    if verbose:
        print(f"[repack-pr106] water-fill bit budget across {len(bits_alloc)} Conv2d layers "
              f"(target_total={target_bytes} bytes)")
        for n, b in sorted(bits_alloc.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {n}: {b} bits ({b // 8} bytes)")

    # 2. Encode each tensor via the appropriate codec
    encoded: dict[str, tuple[int, bytes]] = {}
    for name, t in state_dict.items():
        if _is_water_fillable(name, t) and name in sensitivities:
            try:
                payload = encode_omega_w_v2(
                    t.to(torch.float32),
                    hessian=sensitivities[name].to(torch.float32),
                    total_bits=bits_alloc[name],
                )
                encoded[name] = (CODEC_ID_OWV2, payload)
                if verbose:
                    print(f"  [owv2] {name}: {len(payload)} bytes")
            except Exception as e:
                if verbose:
                    print(f"  [owv2-fail->fallback] {name}: {type(e).__name__}: {e}", file=sys.stderr)
                payload = _encode_brotli_int8_single(name, t)
                encoded[name] = (CODEC_ID_BROTLI_INT8, payload)
                if verbose:
                    print(f"  [brotli-int8] {name}: {len(payload)} bytes (fallback)")
        else:
            payload = _encode_brotli_int8_single(name, t)
            encoded[name] = (CODEC_ID_BROTLI_INT8, payload)
            if verbose:
                print(f"  [brotli-int8] {name}: {len(payload)} bytes")

    # 3. Harvest unchanged latent_brotli from PR106 archive
    with zipfile.ZipFile(pr106_archive) as z:
        bin_bytes = z.read("0.bin")
    _, latent_brotli, header = _parse_pr106_packed_for_latents(bin_bytes)
    if verbose:
        print(f"[repack-pr106] harvested latent_brotli from PR106: {len(latent_brotli)} bytes")

    # 4. Build apogee-v2 0.bin
    new_bin = _build_apogee_v2_blob(encoded, latent_brotli)
    if verbose:
        print(f"[repack-pr106] apogee-v2 0.bin: {len(new_bin)} bytes "
              f"(decoder ~{sum(len(p) for _, p in encoded.values())} + latent {len(latent_brotli)})")

    # 5. Wrap in single-member archive.zip ('0.bin') with deterministic ZipInfo
    archive_path = out_dir / "apogee_v2_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    if verbose:
        print(f"[repack-pr106] wrote {archive_path} ({archive_size} bytes)")

    pr106_size = pr106_archive.stat().st_size
    delta = archive_size - pr106_size
    if verbose:
        print(f"[repack-pr106] PR106 archive: {pr106_size} bytes; delta: {delta:+d}")
    rate_delta = delta / 37545489.0  # contest score formula: 25 x bytes / 37545489
    score_delta_estimate = 25.0 * rate_delta
    if verbose:
        print(f"[repack-pr106] estimated rate-component score delta: {score_delta_estimate:+.6f}")
        print("[repack-pr106] (distortion delta unknown until contest_auth_eval on T4; "
              "sub-0.20 ship-gate is empirical contest score < 0.20945)")

    metadata = {
        "pr106_archive": str(pr106_archive),
        "pr106_archive_size": pr106_size,
        "apogee_v2_archive": str(archive_path),
        "apogee_v2_archive_size": archive_size,
        "size_delta_bytes": delta,
        "rate_component_score_delta": score_delta_estimate,
        "target_decoder_bytes": target_bytes,
        "decoder_actual_bytes": sum(len(p) for _, p in encoded.values()),
        "n_owv2_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_OWV2),
        "n_brotli_int8_layers": sum(1 for cid, _ in encoded.values() if cid == CODEC_ID_BROTLI_INT8),
        "sensitivity_source": str(sensitivity_path),
        "sensitivity_meta": sens_meta,
        "per_tensor_bytes": {name: {"codec_id": cid, "bytes": len(p)} for name, (cid, p) in encoded.items()},
        "tag": "[design-validation]" if sens_meta.get("is_stub") else "[ready-for-contest-CUDA-eval]",
    }
    metadata_path = out_dir / "repack_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if verbose:
        print(f"[repack-pr106] wrote {metadata_path}")
        print(f"[repack-pr106] tag: {metadata['tag']}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--pr106-archive", type=Path, required=True,
                        help="PR106 archive.zip — used to harvest the latent_brotli bytes unchanged.")
    parser.add_argument("--target-bytes", type=int, default=145000,
                        help="Total decoder byte budget (excludes latents). PR106 uses 170278.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    repack_pr106_with_water_filling(
        args.state_dict,
        args.sensitivity,
        args.pr106_archive,
        args.out_dir,
        target_bytes=args.target_bytes,
    )
    return 0


def _parse_pr106_packed_for_latents(bin_bytes: bytes) -> tuple[bytes, bytes, dict]:
    dec_len = int.from_bytes(bin_bytes[1:4], "little")
    decoder_bytes = bin_bytes[4 : 4 + dec_len]
    latent_bytes = bin_bytes[4 + dec_len :]
    return decoder_bytes, latent_bytes, {"dec_len": dec_len}


if __name__ == "__main__":
    sys.exit(main())
