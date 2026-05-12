#!/usr/bin/env python3
"""PR101 lossy int4 with per-output-channel scales — reactivation of
audit criterion #2 for the lossy_int4 lane.

Per the 2026-05-07 audit memo (`feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md`):
- naive PTQ at int4: 37.42% rel_err (measured config not dispatchable)
- QAT at int4 (subagent A): 28.48% rel_err (criterion #1 not dispatchable)
- per-channel scales: NOT TESTED (this tool)

Method:
  For each tensor, derive ONE scale per OUTPUT CHANNEL (axis 0).
  Each row of the weight matrix gets its own fp16 scale.
  Quantize to int4. Pack codes + scales. Brotli. Measure both
  archive bytes and weighted rel_err.

This is the canonical fix for low-bit PTQ collapse. Per CLAUDE.md
"Quantizr intelligence" the 0.33 archive uses per-channel FP4 scales
exactly for this reason.

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

TOOL_NAME = "tools/pr101_lossy_int4_per_channel_scales.py"
SCHEMA_VERSION = "pr101_lossy_int4_per_channel_scales.v1"
INT4_RANGE = 7  # symmetric [-7, +7]
ARCHIVE_OVERHEAD_BYTES = 16_094
EVIDENCE_GRADE = "[CPU-prep empirical reactivation]"
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
BASE_DISPATCH_BLOCKERS = [
    "no_int4_decoder_runtime_built",
    "byte_closed_per_channel_runtime_packet_missing",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_rel_err_not_score_evidence",
]


def per_channel_scales_int4(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-output-channel int4 quant.

    For tensor of shape (C_out, ...), compute one fp16 scale per output
    channel (axis 0). Returns (codes, scales_fp16, recon_fp32).
    """
    if tensor.ndim == 0:
        # Scalar
        v = float(tensor)
        scale = max(abs(v) / INT4_RANGE, 1e-12)
        scale_fp16 = float(np.float16(scale))
        code = int(np.clip(np.round(v / scale_fp16), -INT4_RANGE, +INT4_RANGE))
        recon = float(code * scale_fp16)
        return (np.array([code], dtype=np.int8),
                np.array([scale_fp16], dtype=np.float16),
                np.array([recon], dtype=np.float32))
    # Treat 1D as one channel per element; higher-dimensional tensors use axis 0.
    tensor_re = (
        tensor.reshape(tensor.shape[0], 1)
        if tensor.ndim == 1
        else tensor.reshape(tensor.shape[0], -1)
    )

    n_chan = tensor_re.shape[0]
    codes = np.zeros_like(tensor_re, dtype=np.int8)
    recon = np.zeros_like(tensor_re, dtype=np.float32)
    scales = np.zeros(n_chan, dtype=np.float16)

    for c in range(n_chan):
        row = tensor_re[c]
        abs_max = float(np.abs(row).max())
        if abs_max <= 0.0:
            scales[c] = np.float16(1e-6)
            # codes already zero
            continue
        scale_f = abs_max / INT4_RANGE
        scale_fp16 = float(np.float16(scale_f))
        scales[c] = np.float16(scale_fp16)
        codes_f = row / scale_fp16
        codes[c] = np.clip(np.round(codes_f).astype(np.int8), -INT4_RANGE, +INT4_RANGE)
        recon[c] = (codes[c].astype(np.float32) * scale_fp16).astype(np.float32)

    return codes.flatten(), scales, recon.flatten()


def pack_int4_pairs(codes: np.ndarray) -> bytes:
    if codes.size == 0:
        return b""
    nibbles = (codes.astype(np.int16) + 8).astype(np.uint8)
    if nibbles.size & 1:
        nibbles = np.concatenate([nibbles, np.zeros(1, dtype=np.uint8)])
    packed = (nibbles[0::2] << 4) | (nibbles[1::2] & 0x0F)
    return packed.tobytes()


def encode_tensor_per_channel(tensor: np.ndarray) -> tuple[bytes, dict]:
    if tensor.size == 0:
        return b"", {"n_elements": 0, "n_channels": 0, "scales_bytes": 0, "codes_bytes": 0}
    codes, scales, recon = per_channel_scales_int4(tensor)
    scales_bytes = scales.tobytes()
    codes_bytes = pack_int4_pairs(codes)
    payload = struct.pack("<I", scales.size) + scales_bytes + codes_bytes
    return payload, {
        "n_elements": int(tensor.size),
        "n_channels": int(scales.size),
        "scales_bytes": len(scales_bytes),
        "codes_bytes": len(codes_bytes),
    }


def measure_full(state_dict_path: Path) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    all_payloads: list[bytes] = []
    per_tensor_stats: list[dict] = []
    total_elements = 0
    weighted_rel_err_sum = 0.0
    n_nontrivial_total = 0
    max_p99 = 0.0
    max_max = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        tensor = sd[name].detach().cpu().to(torch.float32).numpy()
        payload, stats = encode_tensor_per_channel(tensor)
        all_payloads.append(payload)

        # Compute rel_err vs original
        codes, scales, recon = per_channel_scales_int4(tensor)
        recon_full = recon[: tensor.size].astype(np.float64)
        orig_full = tensor.flatten().astype(np.float64)
        abs_err = np.abs(recon_full - orig_full)
        eps = 1e-8
        mask = np.abs(orig_full) > eps
        rel_err_pct = np.zeros_like(abs_err)
        rel_err_pct[mask] = 100.0 * abs_err[mask] / np.abs(orig_full[mask])

        n_nontrivial = int(mask.sum())
        rel_err_mean = float(rel_err_pct[mask].mean()) if n_nontrivial > 0 else 0.0
        rel_err_p99 = float(np.percentile(rel_err_pct[mask], 99)) if n_nontrivial > 0 else 0.0
        rel_err_max = float(rel_err_pct[mask].max()) if n_nontrivial > 0 else 0.0

        per_tensor_stats.append({
            "name": name,
            **stats,
            "rel_err_pct_mean": rel_err_mean,
            "rel_err_pct_p99": rel_err_p99,
            "rel_err_pct_max": rel_err_max,
        })
        total_elements += tensor.size
        weighted_rel_err_sum += rel_err_mean * n_nontrivial
        n_nontrivial_total += n_nontrivial
        max_p99 = max(max_p99, rel_err_p99)
        max_max = max(max_max, rel_err_max)

    weighted_avg_rel_err_pct = weighted_rel_err_sum / max(n_nontrivial_total, 1)
    full_blob = b"".join(struct.pack("<I", len(p)) + p for p in all_payloads)
    compressed = brotli.compress(full_blob, quality=11, lgwin=16, lgblock=19)
    archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES

    if weighted_avg_rel_err_pct < 2.0 and max_p99 < 10.0:
        verdict = "CUDA-EVAL-WORTH-TESTING"
        cuda_eval_worth_testing = True
    elif weighted_avg_rel_err_pct < 5.0:
        verdict = "CONDITIONAL"
        cuda_eval_worth_testing = True
    else:
        verdict = "MEASURED_CONFIG_NOT_DISPATCHABLE"
        cuda_eval_worth_testing = False

    dispatch_blockers = list(BASE_DISPATCH_BLOCKERS)
    if weighted_avg_rel_err_pct >= 5.0:
        dispatch_blockers.insert(0, "rel_err_above_5pct_threshold")
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
        "input_state_dict": str(state_dict_path),
        "n_total_elements": total_elements,
        "n_nontrivial_elements": n_nontrivial_total,
        "raw_payload_bytes": len(full_blob),
        "brotli_bytes": len(compressed),
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "weighted_avg_rel_err_pct": weighted_avg_rel_err_pct,
        "max_p99_rel_err_pct": max_p99,
        "max_max_rel_err_pct": max_max,
        "verdict": verdict,
        "dispatch_blockers": dispatch_blockers,
        "per_tensor": per_tensor_stats,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = measure_full(args.state_dict)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_int4_perchannel_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}\n")
    print(f"archive_bytes: {manifest['archive_bytes']:,} B")
    print("  vs naive PTQ block_size=1024: 100,799 B")
    delta_naive = manifest["archive_bytes"] - 100_799
    print(f"  delta vs naive: {delta_naive:+,} B")
    print("\nrel_err pct:")
    print(f"  weighted_avg: {manifest['weighted_avg_rel_err_pct']:.3f}%")
    print(f"  max_p99:      {manifest['max_p99_rel_err_pct']:.3f}%")
    print(f"  max_max:      {manifest['max_max_rel_err_pct']:.3f}%")
    print(f"\nVERDICT: {manifest['verdict']}")
    print(f"  exact-eval ready: {manifest['ready_for_exact_eval_dispatch']}")
    print("  prior naive PTQ: 37.42% rel_err (measured config not dispatchable)")
    print("  prior QAT:       28.48% rel_err (criterion #1 not dispatchable)")
    delta_qat = manifest['weighted_avg_rel_err_pct'] - 28.48
    print(f"  delta vs QAT:    {delta_qat:+.3f}pp")

    if args.output_evidence:
        evidence_row = {
            "technique": "lossy_int4_per_channel_scales",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "empirical_distortion_increase_pct": manifest["weighted_avg_rel_err_pct"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_semantics": manifest["evidence_semantics"],
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
                f"(per-output-channel fp16 scales; audit criterion #2)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "contest_dispatch_verdict": manifest["verdict"],
            "reactivation_criteria_tested": ["per_channel_scales"],
            "reactivation_criteria_remaining": [
                "mixed_precision_int4_int6",
                "GPTQ_calibration",
                "AWQ_calibration",
            ],
            "supersedes_prior_FALSIFIED_tag": False,  # this is per-channel, not the prior naive/QAT lanes
            "dispatch_blockers": manifest["dispatch_blockers"],
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
