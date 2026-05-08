#!/usr/bin/env python3
"""PR101 lossy int4 archive roundtrip distortion test — decides whether
the 100,799 B int4 anchor (block_size=1024) is precise enough to dispatch
for [contest-CUDA] auth eval.

Test:
  1. Quantize PR101 fp32 -> int4 with per-block scales (block_size=1024)
  2. Decode int4 -> fp32 (per-block dequantize)
  3. Compute relative error |q(x) - x| / |x| per element
  4. Report: weighted_avg, p50, p90, p99, max + per-tensor breakdown
  5. Verdict: CUDA-EVAL-WORTH-TESTING if weighted_avg <2% AND p99 <10%
              CONDITIONAL if 2-5% (still worth trying CUDA)
              MEASURED_CONFIG_NOT_DISPATCHABLE if >5%

The threshold (<2% weighted_avg) comes from `feedback_q_faithful_NEVER_
reproduced_quantizr_score_20260505.md`: prior int4 quant attempts went
sour above 2% rel_err per the apogee_intN basin parity findings.

CLAUDE.md compliance: pure CPU + numpy, no scorer load, no contest score
claim. Output is a precision proxy that can recommend more engineering, but it
never marks a lane exact-eval-ready. Actual score requires a byte-closed
runtime packet plus CUDA auth eval on the exact archive bytes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_lossy_int4_roundtrip_test.py"
SCHEMA_VERSION = "pr101_lossy_int4_roundtrip_test.v1"
INT4_RANGE = 7  # symmetric: [-7, +7]
DEFAULT_STATE_DICT_PATH = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt"
)
EVIDENCE_GRADE = "[CPU-prep precision proxy]"
DISPATCH_BLOCKERS_BASE = [
    "no_int4_decoder_runtime_built",
    "byte_closed_int4_runtime_packet_missing",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_rel_err_not_score_evidence",
]


def quantize_dequantize_int4(tensor_flat: np.ndarray, *, block_size: int) -> tuple[np.ndarray, dict]:
    """Per-block int4 quant then dequant. Returns reconstructed fp32 + stats.

    Mirrors the encoder in `pr101_lossy_int4_block_sweep.py` so the
    precision test exactly reflects what the byte-anchor would produce.
    """
    if tensor_flat.size == 0:
        return tensor_flat, {"n_blocks": 0, "n_zero_blocks": 0}
    n_full = tensor_flat.size // block_size
    tail = tensor_flat.size - n_full * block_size
    blocks: list[np.ndarray] = []
    n_zero_blocks = 0
    if n_full > 0:
        body = tensor_flat[: n_full * block_size].reshape(n_full, block_size)
        blocks.extend(body[i] for i in range(n_full))
    if tail:
        blocks.append(tensor_flat[n_full * block_size :])

    recon_parts: list[np.ndarray] = []
    for b in blocks:
        abs_max = float(np.abs(b).max())
        if abs_max <= 0.0:
            recon_parts.append(np.zeros_like(b, dtype=np.float32))
            n_zero_blocks += 1
            continue
        scale_f = abs_max / INT4_RANGE
        scale_fp16 = float(np.float16(scale_f))  # mirrors encoder fp16 storage
        codes_f = b / scale_fp16
        codes = np.clip(np.round(codes_f).astype(np.int8), -INT4_RANGE, +INT4_RANGE)
        recon = (codes.astype(np.float32) * scale_fp16).astype(np.float32)
        recon_parts.append(recon)
    recon_full = np.concatenate(recon_parts).astype(np.float32)
    return recon_full, {"n_blocks": len(blocks), "n_zero_blocks": n_zero_blocks}


def rel_err_stats(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Per-element relative error stats. Skip elements where |orig| < 1e-8 (avoid div-by-zero)."""
    orig_f = orig.flatten().astype(np.float64)
    recon_f = recon.flatten().astype(np.float64)
    abs_err = np.abs(recon_f - orig_f)
    eps = 1e-8
    mask = np.abs(orig_f) > eps
    rel_err_pct = np.zeros_like(abs_err)
    rel_err_pct[mask] = 100.0 * abs_err[mask] / np.abs(orig_f[mask])
    return {
        "n_elements": int(orig.size),
        "n_nontrivial": int(mask.sum()),
        "abs_err_mean": float(abs_err.mean()),
        "abs_err_max": float(abs_err.max()),
        "rel_err_pct_mean": float(rel_err_pct[mask].mean()) if mask.any() else 0.0,
        "rel_err_pct_p50": float(np.percentile(rel_err_pct[mask], 50)) if mask.any() else 0.0,
        "rel_err_pct_p90": float(np.percentile(rel_err_pct[mask], 90)) if mask.any() else 0.0,
        "rel_err_pct_p99": float(np.percentile(rel_err_pct[mask], 99)) if mask.any() else 0.0,
        "rel_err_pct_max": float(rel_err_pct[mask].max()) if mask.any() else 0.0,
    }


def measure_full_roundtrip(state_dict_path: Path, block_size: int) -> dict:
    """Roundtrip-test every tensor in PR101's state_dict at one block_size."""
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    per_tensor: list[dict] = []
    total_elements = 0
    weighted_rel_err_sum = 0.0
    weighted_abs_err_sum = 0.0
    max_p99 = 0.0
    max_max = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        tensor = sd[name].detach().cpu().to(torch.float32).numpy().flatten()
        recon, _block_stats = quantize_dequantize_int4(tensor, block_size=block_size)
        # Truncate recon to original size (in case of tail)
        recon_clipped = recon[: tensor.size]
        stats = rel_err_stats(tensor, recon_clipped)
        per_tensor.append({
            "name": name,
            **stats,
        })
        total_elements += tensor.size
        weighted_rel_err_sum += stats["rel_err_pct_mean"] * stats["n_nontrivial"]
        weighted_abs_err_sum += stats["abs_err_mean"] * tensor.size
        max_p99 = max(max_p99, stats["rel_err_pct_p99"])
        max_max = max(max_max, stats["rel_err_pct_max"])

    n_nontrivial_total = sum(r["n_nontrivial"] for r in per_tensor)
    weighted_avg_rel_err_pct = weighted_rel_err_sum / max(n_nontrivial_total, 1)
    weighted_avg_abs_err = weighted_abs_err_sum / max(total_elements, 1)

    # Verdict — CPU rel_err is a PROXY only. Per CLAUDE.md MPS-falsification
    # rule: NO CPU/MPS measurement may set ready_for_exact_eval_dispatch=True.
    # The verdict here only RECOMMENDS whether a CUDA test is worth running.
    if weighted_avg_rel_err_pct < 2.0 and max_p99 < 10.0:
        verdict = "CUDA-EVAL-WORTH-TESTING"
        verdict_reason = (
            f"weighted_avg {weighted_avg_rel_err_pct:.3f}% < 2.0% "
            f"AND max_p99 {max_p99:.3f}% < 10.0%; CUDA auth eval likely worthwhile"
        )
        cuda_eval_worth_testing = True
    elif weighted_avg_rel_err_pct < 5.0:
        verdict = "CONDITIONAL"
        verdict_reason = (
            f"weighted_avg {weighted_avg_rel_err_pct:.3f}% in [2.0%, 5.0%); "
            f"borderline — CUDA test recommended but expect ~10-30% score regression"
        )
        cuda_eval_worth_testing = True
    else:
        verdict = "MEASURED_CONFIG_NOT_DISPATCHABLE"
        verdict_reason = (
            f"weighted_avg {weighted_avg_rel_err_pct:.3f}% >= 5.0%; "
            f"reconstruction too lossy for this measured config; "
            f"family stays DEFERRED-pending-research per audit"
        )
        cuda_eval_worth_testing = False

    # ready_for_exact_eval_dispatch is ALWAYS False from CPU/MPS evidence.
    # Only [contest-CUDA] auth eval can flip it to True.
    dispatch_blockers = list(DISPATCH_BLOCKERS_BASE)
    if not cuda_eval_worth_testing:
        dispatch_blockers.insert(0, "rel_err_above_5pct_threshold")

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": "cpu_roundtrip_rel_err_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "cuda_eval_worth_testing": cuda_eval_worth_testing,
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "block_size": block_size,
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "n_total_elements": total_elements,
        "n_nontrivial_elements": n_nontrivial_total,
        "weighted_avg_rel_err_pct": weighted_avg_rel_err_pct,
        "weighted_avg_abs_err": weighted_avg_abs_err,
        "max_p99_rel_err_pct": max_p99,
        "max_max_rel_err_pct": max_max,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "dispatch_blockers": dispatch_blockers,
        "per_tensor": per_tensor,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT_PATH)
    p.add_argument("--block-size", type=int, default=1024,
                   help="block_size that won the int4 sweep (default: 1024)")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--also-block-sizes", type=int, nargs="*", default=None,
                   help="Optional additional block sizes to compare precision")
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    block_sizes = [args.block_size]
    if args.also_block_sizes:
        block_sizes.extend(args.also_block_sizes)

    runs: list[dict] = []
    for bs in block_sizes:
        manifest = measure_full_roundtrip(args.state_dict, bs)
        runs.append(manifest)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_int4_roundtrip_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"runs": runs, "tool": TOOL_NAME, "schema": SCHEMA_VERSION}, indent=2),
        encoding="utf-8",
    )

    print(f"\nmanifest: {args.output_json}\n")
    print(f"  block_size | weighted_avg_rel_err | max_p99 | max_max | verdict")
    for r in runs:
        print(f"  {r['block_size']:>10} | {r['weighted_avg_rel_err_pct']:>18.4f}% | "
              f"{r['max_p99_rel_err_pct']:>6.3f}% | {r['max_max_rel_err_pct']:>6.3f}% | "
              f"{r['verdict']}")
    print()
    primary = runs[0]
    print(f"PRIMARY VERDICT (block_size={primary['block_size']}): {primary['verdict']}")
    print(f"  reason: {primary['verdict_reason']}")
    if primary["cuda_eval_worth_testing"]:
        print(f"\nNEXT: build a byte-closed int4 runtime packet before any CUDA auth eval.")
    else:
        print(f"\nNEXT: do NOT dispatch CUDA. Higher precision int5/int6 may help; "
              f"or fall back to int8 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
