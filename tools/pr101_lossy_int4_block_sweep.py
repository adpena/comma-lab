#!/usr/bin/env python3
"""PR101 lossy int4 quantization block-size sweep — anchors the
``lossy_int4_quantization`` row in cathedral_autopilot's catalog.

Per-block symmetric int4 quantization: each block of N elements gets one
fp16 scale + N int4 codes (4 bits per element). Concatenate all blocks +
scales, brotli, measure. Sweep block sizes {32, 64, 128, 256, 512, 1024}.

The current catalog row predicts 105,440 B at block_size=64. Empirical
sweep confirms or falsifies that anchor. Emits a JSONL evidence row that
``cathedral_autopilot evidence-update`` ingests.

CLAUDE.md compliance: pure CPU + numpy + brotli; no scorer load; no
contest score claims; output tagged ``[CPU-prep empirical]`` (the
quantization is real, byte counts are real, but no inflation/score is
performed — score lives downstream).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_lossy_int4_block_sweep.py"
SCHEMA_VERSION = "pr101_lossy_int4_block_sweep.v2"
INT4_RANGE = 7  # symmetric: [-7, +7], 15 levels (with zero) = int4
ARCHIVE_OVERHEAD_BYTES = 16_094  # PR101's archive zip overhead (constant)
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
PREDICTED_LOSSY_INT4_ARCHIVE_BYTES = 105_440
EVIDENCE_GRADE = "empirical"
EVIDENCE_MARKER = "[CPU-prep empirical]"
EVIDENCE_SEMANTICS = "cpu_lossy_int4_quantization_byte_anchor_no_decode_no_score"
DISPATCH_BLOCKERS = [
    "lossy_weight_quantization_not_decoded_or_scored",
    "no_runtime_decoder_packet_built",
    "no_archive_substitution_performed",
    "component_distortion_unknown",
    "missing_exact_cuda_auth_eval",
]
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)


def repo_relative(path: Path) -> str:
    """Return a stable repo-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quantize_block_int4(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-block symmetric int4 quant. Returns (codes [-7,+7], scale fp16)."""
    abs_max = float(np.abs(block).max())
    if abs_max <= 0.0:
        scale = np.float16(1e-6)
        codes = np.zeros_like(block, dtype=np.int8)
        return codes, float(scale)
    scale_f = abs_max / INT4_RANGE
    # Round-half-to-even via banker's rounding to match deterministic decoders.
    codes_f = block / scale_f
    codes = np.clip(np.round(codes_f).astype(np.int8), -INT4_RANGE, +INT4_RANGE)
    scale_fp16 = np.float16(scale_f)
    return codes.astype(np.int8), float(scale_fp16)


def pack_int4_pairs(codes: np.ndarray) -> bytes:
    """Pack int4 codes (in -7..+7) into 4-bit nibbles, two per byte.

    Codes are biased by +8 to fit u4 [0..15] before packing; the unpacker
    subtracts 8 to recover the signed value.
    """
    if codes.size == 0:
        return b""
    nibbles = (codes.astype(np.int16) + 8).astype(np.uint8)
    if nibbles.size & 1:
        nibbles = np.concatenate([nibbles, np.zeros(1, dtype=np.uint8)])
    packed = (nibbles[0::2] << 4) | (nibbles[1::2] & 0x0F)
    return packed.tobytes()


def encode_tensor(tensor_flat: np.ndarray, *, block_size: int) -> tuple[bytes, dict]:
    """Encode a flat numpy tensor block-by-block. Returns (payload, stats)."""
    if tensor_flat.size == 0:
        return b"", {"n_elements": 0, "n_blocks": 0, "scales_bytes": 0, "codes_bytes": 0}
    n_full_blocks = tensor_flat.size // block_size
    tail_size = tensor_flat.size - n_full_blocks * block_size
    n_blocks = n_full_blocks + (1 if tail_size else 0)
    all_codes: list[np.ndarray] = []
    scales: list[float] = []
    for b in range(n_full_blocks):
        block = tensor_flat[b * block_size : (b + 1) * block_size]
        codes, scale = quantize_block_int4(block)
        all_codes.append(codes)
        scales.append(scale)
    if tail_size:
        block = tensor_flat[n_full_blocks * block_size :]
        codes, scale = quantize_block_int4(block)
        all_codes.append(codes)
        scales.append(scale)
    codes_concat = np.concatenate(all_codes)
    scales_arr = np.array(scales, dtype=np.float16)
    codes_packed = pack_int4_pairs(codes_concat)
    scales_bytes = scales_arr.tobytes()
    payload = scales_bytes + codes_packed
    return payload, {
        "n_elements": int(tensor_flat.size),
        "n_blocks": n_blocks,
        "scales_bytes": len(scales_bytes),
        "codes_bytes": len(codes_packed),
    }


def sweep_block_sizes(
    state_dict_path: Path,
    block_sizes: list[int],
    *,
    n_quant: int = 0,  # unused: int4 ignores N_QUANT
) -> dict:
    """Quantize the full state_dict at each block_size, brotli, return stats."""
    import torch

    input_sha256 = sha256_file(state_dict_path)
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    rows: list[dict] = []
    for bs in block_sizes:
        all_payloads: list[bytes] = []
        per_tensor_stats: list[dict] = []
        total_elements = 0
        for name, _shape in FIXED_STATE_SCHEMA:
            if name not in sd:
                raise SystemExit(f"state_dict missing {name!r}")
            tensor = sd[name].detach().cpu().to(torch.float32).numpy().flatten()
            payload, stats = encode_tensor(tensor, block_size=bs)
            all_payloads.append(payload)
            per_tensor_stats.append({"name": name, **stats})
            total_elements += stats["n_elements"]
        full_payload = b"".join(all_payloads)
        # Brotli at the same params our autopilot brotli_optuna_default uses
        compressed = brotli.compress(
            full_payload, quality=11, lgwin=16, lgblock=19, mode=brotli.MODE_GENERIC
        )
        archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES
        rows.append({
            "block_size": bs,
            "raw_payload_bytes": len(full_payload),
            "brotli_bytes": len(compressed),
            "archive_bytes": archive_bytes,
            "compression_ratio": len(compressed) / max(len(full_payload), 1),
            "n_elements": total_elements,
            "bits_per_element": (len(compressed) * 8) / max(total_elements, 1),
            "per_tensor": per_tensor_stats,
        })

    rows.sort(key=lambda r: r["archive_bytes"])
    best = rows[0]
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_MARKER,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "dispatch_blockers": DISPATCH_BLOCKERS,
        "input_state_dict": repo_relative(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "block_sizes_swept": block_sizes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "best_block_size": best["block_size"],
        "best_archive_bytes": best["archive_bytes"],
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "delta_vs_brotli_optuna_archive_bytes": (
            best["archive_bytes"] - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        ),
        "catalog_prediction_archive_bytes": PREDICTED_LOSSY_INT4_ARCHIVE_BYTES,
        "delta_vs_catalog_prediction_bytes": (
            best["archive_bytes"] - PREDICTED_LOSSY_INT4_ARCHIVE_BYTES
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--block-sizes", type=int, nargs="+",
                   default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--output-json", type=Path, default=None,
                   help="Manifest path; defaults to reports/raw/.../manifest.json")
    p.add_argument("--output-evidence", type=Path, default=None,
                   help="JSONL evidence row for cathedral_autopilot")
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = sweep_block_sizes(args.state_dict, args.block_sizes)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_int4_sweep_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {args.output_json}")

    print(
        f"\nbest block_size: {manifest['best_block_size']} "
        f"(archive={manifest['best_archive_bytes']:,} B)"
    )
    print(f"  vs cathedral_autopilot prediction: {PREDICTED_LOSSY_INT4_ARCHIVE_BYTES:,} B")
    delta = manifest["delta_vs_catalog_prediction_bytes"]
    print(f"  delta: {delta:+,} B "
          f"({'BEAT' if delta < -1000 else 'TIED' if abs(delta) <= 1000 else 'MISSED'} prediction)")
    print(
        "  evidence: CPU byte anchor only; no decoder packet, no score claim, "
        "not promotable"
    )
    print()
    print("  block_size | archive_bytes | bits/elem")
    for r in manifest["rows"]:
        print(f"  {r['block_size']:>10} | {r['archive_bytes']:>13,} | "
              f"{r['bits_per_element']:.4f}")

    if args.output_evidence:
        evidence_row = {
            "technique": "lossy_int4_quantization",
            "empirical_archive_bytes": manifest["best_archive_bytes"],
            "empirical_d_seg": None,
            "empirical_d_pose": None,
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_MARKER,
            "evidence_semantics": EVIDENCE_SEMANTICS,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "dispatch_blockers": DISPATCH_BLOCKERS,
            "source": (
                f"{EVIDENCE_MARKER} {repo_relative(args.output_json)} "
                f"(block_size={manifest['best_block_size']})"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
