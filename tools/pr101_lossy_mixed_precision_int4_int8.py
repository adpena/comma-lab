#!/usr/bin/env python3
"""PR101 lossy mixed-precision int4/int6/int8 — audit criterion #3 for
the lossy_int4 lane.

Per the 2026-05-07 audit memo (`feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md`):
- naive PTQ at int4: 37.42% rel_err (measured config not dispatchable)
- QAT at int4: 28.48% rel_err (criterion #1 not dispatchable)
- per-channel scales: 30.41% rel_err (criterion #2 not dispatchable)
- mixed-precision int4/int6/int8: NOT TESTED (this tool — criterion #3)

Method (per-tensor sensitivity-driven bit allocation):
  1. Quantize each tensor independently at int4, int6, int8
  2. Compute per-tensor rel_err for each precision
  3. Weighted greedy assignment: minimize total bytes subject to
     weighted_avg_rel_err < target_pct
  4. Pack: scales + bit-width header per tensor + variable-bit-width payload + brotli

The intuition: not all tensors need the same precision. Layers near
the input/output that touch raw pixel data may need int8, while
deeper feature transforms may tolerate int4.

CLAUDE.md compliance: pure CPU + numpy + brotli; no scorer load; no
contest score claim; output tagged ``[CPU-prep empirical reactivation]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_lossy_mixed_precision_int4_int8.py"
SCHEMA_VERSION = "pr101_lossy_mixed_precision_int4_int8.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
PR101_BROTLI_BASELINE_BYTES = 178_144
TENSOR_HEADER_BYTES = 12  # n_bits, n_scales, n_codes uint32 fields.
EVIDENCE_GRADE = "[CPU-prep empirical reactivation]"
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
BASE_DISPATCH_BLOCKERS = [
    "no_int4_decoder_runtime_built",
    "byte_closed_mixed_precision_runtime_packet_missing",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_rel_err_not_score_evidence",
]
UNTESTED_REACTIVATION_CRITERIA = ["GPTQ_calibration", "AWQ_calibration"]


def quantize_at_n_bits(tensor_flat: np.ndarray, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-block quantization at n_bits. Block size = 1024 (best from prior sweep).

    Returns (codes_packed, scales_fp16).
    """
    block_size = 1024
    abs_range = (1 << (n_bits - 1)) - 1  # int4: 7, int6: 31, int8: 127
    n_full = tensor_flat.size // block_size
    tail = tensor_flat.size - n_full * block_size
    blocks = []
    if n_full > 0:
        for i in range(n_full):
            blocks.append(tensor_flat[i * block_size : (i + 1) * block_size])
    if tail:
        blocks.append(tensor_flat[n_full * block_size :])

    all_codes: list[np.ndarray] = []
    all_recon: list[np.ndarray] = []
    scales = np.zeros(len(blocks), dtype=np.float16)
    for i, block in enumerate(blocks):
        abs_max = float(np.abs(block).max())
        if abs_max <= 0.0:
            scales[i] = np.float16(1e-6)
            all_codes.append(np.zeros_like(block, dtype=np.int8))
            all_recon.append(np.zeros_like(block, dtype=np.float32))
            continue
        scale_f = abs_max / abs_range
        scale_fp16 = float(np.float16(scale_f))
        scales[i] = np.float16(scale_fp16)
        codes = np.clip(np.round(block / scale_fp16).astype(np.int16), -abs_range, +abs_range)
        all_codes.append(codes.astype(np.int8 if n_bits <= 8 else np.int16))
        all_recon.append(codes.astype(np.float32) * scale_fp16)
    codes_concat = np.concatenate(all_codes)
    recon_concat = np.concatenate(all_recon).astype(np.float32)
    # Truncate to original size
    codes_concat = codes_concat[: tensor_flat.size]
    recon_concat = recon_concat[: tensor_flat.size]
    return codes_concat, scales, recon_concat


def per_tensor_rel_err_stats(orig: np.ndarray, recon: np.ndarray) -> dict[str, float | int]:
    """Elementwise relative-error stats over nonzero original elements.

    CPU relative error is a proxy for weight perturbation only; it is not a
    contest distortion, score, promotion, rank, or kill signal.
    """
    eps = 1e-8
    orig_f = orig.flatten().astype(np.float64)
    recon_f = recon.flatten().astype(np.float64)
    abs_err = np.abs(recon_f - orig_f)
    mask = np.abs(orig_f) > eps
    if not mask.any():
        return {"rel_err_pct_mean": 0.0, "n_nontrivial": 0}
    rel_err = 100.0 * abs_err[mask] / np.abs(orig_f[mask])  # REL_ERR_NON_CANONICAL_OK: per-element L1 percentage for PR101 int4/int8 mixed-precision sweep; not allocator-fed
    return {
        "rel_err_pct_mean": float(rel_err.mean()),
        "n_nontrivial": int(mask.sum()),
    }


def per_tensor_rel_err(orig: np.ndarray, recon: np.ndarray) -> float:
    return float(per_tensor_rel_err_stats(orig, recon)["rel_err_pct_mean"])


def encoded_bytes_for_tensor(n_elements: int, n_blocks: int, n_bits: int) -> int:
    """Exact uncompressed bytes for this tool's per-tensor wire record."""
    scales_bytes = n_blocks * 2  # fp16 = 2 bytes
    if n_bits == 4:
        codes_bytes = (n_elements + 1) // 2  # 4 bits per code, 2 codes per byte
    elif n_bits == 6:
        # This tool pads to 4-code groups, then emits 3 bytes per group.
        codes_bytes = ((n_elements + 3) // 4) * 3
    elif n_bits == 8:
        codes_bytes = n_elements
    else:
        codes_bytes = (n_elements * n_bits + 7) // 8
    return TENSOR_HEADER_BYTES + scales_bytes + codes_bytes


def classify_cpu_proxy_candidate(
    *,
    weighted_avg_rel_err_pct: float,
    archive_bytes: int,
    target_rel_err_pct: float,
) -> tuple[str, bool, list[str]]:
    """Fail-closed CPU proxy classification.

    A local byte/rel_err point can recommend more engineering, but exact CUDA
    auth eval on a byte-closed runtime packet is required before any score,
    promotion, rank, or kill semantics. A point that is larger than the PR101
    brotli baseline and also lossy is Pareto-dominated and should not route to
    CUDA spend.
    """
    blockers = list(BASE_DISPATCH_BLOCKERS)
    beats_baseline = archive_bytes < PR101_BROTLI_BASELINE_BYTES
    below_target = weighted_avg_rel_err_pct < target_rel_err_pct
    if not beats_baseline:
        blockers.insert(0, "archive_bytes_not_below_pr101_brotli_baseline")
    if not below_target:
        blockers.insert(0, "rel_err_above_target_threshold")

    if not beats_baseline:
        return "MEASURED_CONFIG_DOMINATED_BY_PR101_BROTLI_BASELINE", False, blockers
    if weighted_avg_rel_err_pct < 2.0:
        return "CUDA-EVAL-WORTH-TESTING", True, blockers
    if below_target:
        return "CONDITIONAL-CUDA-EVAL-WORTH-TESTING", True, blockers
    return "MEASURED_CONFIG_NOT_DISPATCHABLE", False, blockers


def assign_per_tensor_bits(
    per_tensor_data: list[tuple[str, np.ndarray]],
    target_rel_err_pct: float,
) -> dict:
    """Greedy: start all int4, upgrade highest-sensitivity tensors to int6 then int8
    until weighted_avg_rel_err crosses below target."""
    block_size = 1024

    # Compute rel_err for each (tensor, precision) combination
    options: list[dict] = []
    for name, tensor_flat in per_tensor_data:
        n = tensor_flat.size
        n_blocks = (n + block_size - 1) // block_size
        opts = []
        n_nontrivial = 0
        for n_bits in (4, 6, 8):
            _codes, _scales, recon = quantize_at_n_bits(tensor_flat, n_bits)
            err_stats = per_tensor_rel_err_stats(tensor_flat, recon)
            err = float(err_stats["rel_err_pct_mean"])
            n_nontrivial = int(err_stats["n_nontrivial"])
            bytes_ = encoded_bytes_for_tensor(n, n_blocks, n_bits)
            opts.append({"n_bits": n_bits, "rel_err_pct": err, "bytes": bytes_})
        options.append({
            "name": name,
            "n_elements": n,
            "n_nontrivial": n_nontrivial,
            "n_blocks": n_blocks,
            "options": opts,
        })

    # Greedy: start at int4 for all; total weighted_avg_rel_err
    chosen = {opt["name"]: 4 for opt in options}
    n_nontrivial_total = sum(opt["n_nontrivial"] for opt in options)

    def compute_weighted_err(chosen: dict) -> float:
        s = 0.0
        for opt in options:
            err = next(o["rel_err_pct"] for o in opt["options"] if o["n_bits"] == chosen[opt["name"]])
            s += err * opt["n_nontrivial"]
        return s / max(n_nontrivial_total, 1)

    # Iteratively upgrade the tensor with the largest current rel_err contribution
    for _step in range(2 * len(options) + 1):
        cur_err = compute_weighted_err(chosen)
        if cur_err < target_rel_err_pct:
            break
        # Find best upgrade: which tensor's int4→int6 or int6→int8 reduces weighted err
        # the most per byte added?
        best_gain = None
        best_name = None
        best_new_bits = None
        for opt in options:
            cur = chosen[opt["name"]]
            if cur >= 8:
                continue
            new_bits = 6 if cur == 4 else 8
            cur_opt = next(o for o in opt["options"] if o["n_bits"] == cur)
            new_opt = next(o for o in opt["options"] if o["n_bits"] == new_bits)
            err_delta = (
                cur_opt["rel_err_pct"] - new_opt["rel_err_pct"]
            ) * opt["n_nontrivial"]
            byte_delta = new_opt["bytes"] - cur_opt["bytes"]
            gain = float("inf") if byte_delta <= 0 else err_delta / byte_delta
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_name = opt["name"]
                best_new_bits = new_bits
        if best_name is None:
            break
        chosen[best_name] = best_new_bits

    # Compute final stats
    total_payload_bytes = 0
    total_err_sum = 0.0
    per_tensor_final = []
    for opt in options:
        bits = chosen[opt["name"]]
        chosen_opt = next(o for o in opt["options"] if o["n_bits"] == bits)
        total_payload_bytes += chosen_opt["bytes"]
        total_err_sum += chosen_opt["rel_err_pct"] * opt["n_nontrivial"]
        per_tensor_final.append({
            "name": opt["name"],
            "n_elements": opt["n_elements"],
            "n_nontrivial": opt["n_nontrivial"],
            "n_blocks": opt["n_blocks"],
            "chosen_n_bits": bits,
            "rel_err_pct": chosen_opt["rel_err_pct"],
            "bytes": chosen_opt["bytes"],
        })
    weighted_avg_rel_err_pct = total_err_sum / max(n_nontrivial_total, 1)

    n_int4 = sum(1 for v in chosen.values() if v == 4)
    n_int6 = sum(1 for v in chosen.values() if v == 6)
    n_int8 = sum(1 for v in chosen.values() if v == 8)

    return {
        "weighted_avg_rel_err_pct": weighted_avg_rel_err_pct,
        "raw_payload_bytes_estimate": total_payload_bytes,
        "n_nontrivial_elements": n_nontrivial_total,
        "n_tensors_int4": n_int4,
        "n_tensors_int6": n_int6,
        "n_tensors_int8": n_int8,
        "per_tensor": per_tensor_final,
        "chosen_assignment": dict(chosen),
    }


def measure_full(state_dict_path: Path, target_rel_err_pct: float) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    per_tensor_data: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        tensor = sd[name].detach().cpu().to(torch.float32).numpy().flatten()
        per_tensor_data.append((name, tensor))

    assignment = assign_per_tensor_bits(per_tensor_data, target_rel_err_pct)

    # For final byte estimate, brotli over the assigned-precision payload
    # We approximate without actually packing — use raw bytes + ~30% brotli compression
    # Actually let's pack: int4 codes + int6 codes (packed in nibbles+halves) + int8 codes
    full_blob = bytearray()
    for opt_row in assignment["per_tensor"]:
        n_bits = opt_row["chosen_n_bits"]
        name = opt_row["name"]
        tensor = next(t for n, t in per_tensor_data if n == name)
        codes, scales, _recon = quantize_at_n_bits(tensor, n_bits)
        # Pack header: name length, n_bits, n_blocks, then scales + codes
        full_blob += struct.pack("<I", n_bits)
        full_blob += struct.pack("<I", scales.size)
        full_blob += scales.tobytes()
        if n_bits == 4:
            biased = (codes.astype(np.int16) + 8).astype(np.uint8)
            if biased.size & 1:
                biased = np.concatenate([biased, np.zeros(1, dtype=np.uint8)])
            packed = (biased[0::2] << 4) | (biased[1::2] & 0x0F)
            full_blob += struct.pack("<I", codes.size)
            full_blob += packed.tobytes()
        elif n_bits == 6:
            # Pack 4 codes into 3 bytes
            biased = (codes.astype(np.int16) + 32).astype(np.uint8)
            n_pad = (4 - biased.size % 4) % 4
            if n_pad:
                biased = np.concatenate([biased, np.zeros(n_pad, dtype=np.uint8)])
            packed = bytearray()
            for i in range(0, biased.size, 4):
                a, b, c, d = biased[i:i+4]
                packed.append(((a & 0x3F) << 2) | ((b >> 4) & 0x03))
                packed.append(((b & 0x0F) << 4) | ((c >> 2) & 0x0F))
                packed.append(((c & 0x03) << 6) | (d & 0x3F))
            full_blob += struct.pack("<I", codes.size)
            full_blob += bytes(packed)
        elif n_bits == 8:
            full_blob += struct.pack("<I", codes.size)
            full_blob += codes.astype(np.int8).tobytes()

    compressed = brotli.compress(bytes(full_blob), quality=11, lgwin=16, lgblock=19)
    archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES

    if assignment["raw_payload_bytes_estimate"] != len(full_blob):
        raise RuntimeError(
            "mixed-precision byte accounting drift: "
            f"estimate={assignment['raw_payload_bytes_estimate']} "
            f"packed={len(full_blob)}"
        )

    verdict, cuda_eval_worth_testing, dispatch_blockers = classify_cpu_proxy_candidate(
        weighted_avg_rel_err_pct=assignment["weighted_avg_rel_err_pct"],
        archive_bytes=archive_bytes,
        target_rel_err_pct=target_rel_err_pct,
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": "cpu_roundtrip_rel_err_and_byte_anchor_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "cuda_eval_worth_testing": cuda_eval_worth_testing,
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "reactivation_criteria_tested": ["mixed_precision_int4_int6_int8"],
        "reactivation_criteria_remaining": list(UNTESTED_REACTIVATION_CRITERIA),
        "input_state_dict": str(state_dict_path),
        "target_rel_err_pct": target_rel_err_pct,
        "rel_err_definition": (
            "mean(abs(recon_fp32-orig_fp32)/abs(orig_fp32))*100 over elements "
            "with abs(orig_fp32)>1e-8; tensor means weighted by nontrivial "
            "element count; CPU proxy only, not contest distortion"
        ),
        "raw_payload_bytes_packed": len(full_blob),
        "raw_payload_bytes_estimate": assignment["raw_payload_bytes_estimate"],
        "raw_payload_bytes_estimate_matches_packed": True,
        "brotli_bytes": len(compressed),
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "weighted_avg_rel_err_pct": assignment["weighted_avg_rel_err_pct"],
        "n_nontrivial_elements": assignment["n_nontrivial_elements"],
        "n_tensors_int4": assignment["n_tensors_int4"],
        "n_tensors_int6": assignment["n_tensors_int6"],
        "n_tensors_int8": assignment["n_tensors_int8"],
        "verdict": verdict,
        "dispatch_blockers": dispatch_blockers,
        "per_tensor": assignment["per_tensor"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--target-rel-err-pct", type=float, default=5.0,
                   help="Target weighted-avg rel_err threshold (default: 5.0)")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = measure_full(args.state_dict, args.target_rel_err_pct)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_int4_mixed_precision_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}\n")
    print(f"target_rel_err_pct: {args.target_rel_err_pct}%")
    print(f"weighted_avg_rel_err_pct: {manifest['weighted_avg_rel_err_pct']:.3f}%")
    print(f"\narchive_bytes: {manifest['archive_bytes']:,} B")
    print("  vs brotli baseline: 178,144 B")
    delta = manifest["archive_bytes"] - 178_144
    print(f"  delta: {delta:+,} B "
          f"({'BEAT' if delta < 0 else 'TIES' if abs(delta) < 100 else 'LOSES'} brotli)")
    print(f"\ntensor allocation: {manifest['n_tensors_int4']} int4, "
          f"{manifest['n_tensors_int6']} int6, {manifest['n_tensors_int8']} int8")
    print(f"\nVERDICT: {manifest['verdict']}")
    print(f"  exact-eval ready:  {manifest['ready_for_exact_eval_dispatch']}")
    print("  prior naive PTQ:   37.42% rel_err (measured config not dispatchable)")
    print("  prior QAT:         28.48% rel_err (criterion #1 not dispatchable)")
    print("  prior per-channel: 30.41% rel_err (criterion #2 not dispatchable)")
    delta_qat = manifest['weighted_avg_rel_err_pct'] - 28.48
    print(f"  delta vs QAT:      {delta_qat:+.3f}pp")

    if args.output_evidence:
        evidence_row = {
            "technique": "lossy_int4_mixed_precision",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "empirical_distortion_increase_pct": manifest["weighted_avg_rel_err_pct"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_semantics": manifest["evidence_semantics"],
            "rel_err_definition": manifest["rel_err_definition"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": manifest["cuda_eval_worth_testing"],
            "family_falsified": False,
            "falsification_scope": "measured_configuration_only",
            "source": (
                f"[CPU-prep empirical reactivation] {args.output_json} "
                f"(mixed-precision int4/int6/int8; audit criterion #3; greedy assignment)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "contest_dispatch_verdict": manifest["verdict"],
            "reactivation_criteria_tested": ["mixed_precision_int4_int6_int8"],
            "reactivation_criteria_remaining": list(UNTESTED_REACTIVATION_CRITERIA),
            "tensor_allocation": {
                "int4": manifest["n_tensors_int4"],
                "int6": manifest["n_tensors_int6"],
                "int8": manifest["n_tensors_int8"],
            },
            "dispatch_blockers": manifest["dispatch_blockers"],
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
