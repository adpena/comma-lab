#!/usr/bin/env python3
"""PR101 analytical lossy coarsening — byte/roundtrip proxy against the PR101
brotli baseline at controlled rel_err.

Finding (2026-05-08)
--------------------
A simple per-tensor quantization-step (K) sweep, where each PR101 tensor's
INT8 symbols are rounded to the nearest multiple of K then brotli-compressed,
is byte-lower than the canonical PR101 brotli baseline (178,144 B) by 18-22 KB
at rel_err under 5%:

  budget=0.05: rel_err=0.0386, archive=156,344 B (-21,800 B)  [byte proxy]
  budget=0.03: rel_err=0.0149, archive=170,200 B  (-7,944 B)
  budget=0.02: rel_err=0.0019, archive=176,990 B  (-1,154 B)

Prior CompressAI smoke experiments
(``tools/pr101_compressai_balle_hyperprior_full.py`` rel_err 0.98;
``tools/pr101_compressai_factorized_prior.py`` rel_err 0.67;
``tools/pr101_neural_weight_codec_NWC.py`` rel_err 0.92) did not reach
rel_err < 5% in their measured configs. That is not a family kill; it only
defines the local R(D) proxy they must beat.

1. Per-tensor scale heterogeneity (each tensor has its own dynamic range) is
   exactly what the per-tensor K naturally exploits — neural codecs need to
   re-learn this per-tensor structure at high model-overhead cost.
2. PR101 symbols appear near-iid in the measured proxy streams (joint-entropy floor 148-162 KB
   per ``feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md``);
   this narrows the measured lossless-coder headroom without proving a
   neural-codec impossibility.
3. The lossless brotli baseline is near the measured marginal-entropy proxy;
   the observed byte headroom here is lossy and must be validated by runtime
   decode plus exact CUDA auth eval before it can affect score.

This tool is the analytical foundation: it finds the best per-tensor K
schedule for a given rel_err budget and emits the archive bytes. It serves
as the credible point on the R(D) curve that learned codecs must exceed to
be useful for PR101.

Wire format
-----------
- 28 × 1 byte: per-tensor K (uint8)
- brotli(concat(rounded_symbols_int8))
- canonical PR101 archive overhead 16,094 B

Total: payload + 28 + 16,094

CLAUDE.md compliance: pure-CPU analytical, evidence tagged
``[MPS-research-signal]`` per the user mandate (no scorer-load, no score
claim — just byte + roundtrip-rel_err). Any score, promotion, rank, or kill
requires exact CUDA auth eval on a byte-closed runtime packet.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_lossy_coarsening_analytical.py"
SCHEMA_VERSION = "pr101_lossy_coarsening_analytical.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
PR101_BROTLI_BASELINE_BYTES = 178_144
N_TENSORS = len(FIXED_STATE_SCHEMA)
DISPATCH_BLOCKERS = [
    "missing_exact_cuda_auth_eval",
    "missing_exact_cuda_auth_eval_on_lossy_decoder",
    "no_runtime_dequantize_path_built",
    "byte_closed_lossy_coarsening_runtime_packet_missing",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "proxy_rel_err_not_score_evidence",
]


@dataclass
class TensorBlob:
    name: str
    raw: np.ndarray  # int32 [n], range [-127, 127]


def collect_tensors(state_dict_path: Path) -> list[TensorBlob]:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    out: list[TensorBlob] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        out.append(TensorBlob(name=name, raw=qt.q_i8.astype(np.int32).flatten()))
    return out


def find_best_K_for_tensor(symbols: np.ndarray, budget: float) -> tuple[int, float]:
    """Find largest K such that per-tensor rel_err <= budget.

    Returns (best_K, achieved_rel_err).
    """
    abs_sum = float(np.abs(symbols).astype(np.float64).sum())
    if abs_sum < 1e-9:
        return 1, 0.0
    best_K = 1
    best_re = 0.0
    for K in range(1, 256):
        rounded = np.round(symbols / K) * K
        err = float(np.abs(rounded - symbols).astype(np.float64).sum())
        re = err / abs_sum
        if re <= budget:
            best_K = K
            best_re = re
        else:
            break
    return best_K, best_re


def encode_with_per_tensor_K(
    tensors: list[TensorBlob], Ks: list[int], brotli_quality: int = 11,
) -> dict:
    """Encode all tensors with per-tensor K, return bytes + recon stats."""
    abs_orig_total = 0.0
    abs_err_total = 0.0
    rounded_chunks: list[np.ndarray] = []
    for tb, K in zip(tensors, Ks, strict=True):
        rounded = np.round(tb.raw / K) * K
        err = float(np.abs(rounded - tb.raw).astype(np.float64).sum())
        abs_err_total += err
        abs_orig_total += float(np.abs(tb.raw).astype(np.float64).sum())
        rounded_chunks.append(rounded.clip(-127, 127).astype(np.int8))

    flat_bytes = np.concatenate(rounded_chunks).tobytes()
    payload_brotli = brotli.compress(
        flat_bytes, quality=brotli_quality, lgwin=22, lgblock=24,
    )
    side_info = bytes(Ks)  # 28 bytes
    archive_bytes = len(payload_brotli) + len(side_info) + ARCHIVE_OVERHEAD_BYTES
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0
    return {
        "payload_brotli_bytes": len(payload_brotli),
        "side_info_bytes": len(side_info),
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "rel_err": rel_err,
        "abs_err_sum": abs_err_total,
        "abs_orig_sum": abs_orig_total,
        "Ks": list(Ks),
    }


def encode_with_uniform_K(
    tensors: list[TensorBlob], K: int, brotli_quality: int = 11,
) -> dict:
    """Single K applied to every tensor (analytical baseline)."""
    return encode_with_per_tensor_K(tensors, [K] * len(tensors), brotli_quality)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="0.005,0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated per-tensor rel_err budgets.",
    )
    p.add_argument(
        "--uniform-Ks",
        type=str,
        default="1,2,3,5,7,10",
        help="Comma-separated uniform-K baselines to also report.",
    )
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--evidence-jsonl",
        type=Path,
        default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl",
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")
    if args.output_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / f"reports/raw/pr101_lossy_coarsening_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lossy-coarsen] state_dict: {args.state_dict}")
    print(f"[lossy-coarsen] output:     {args.output_dir}")

    tensors = collect_tensors(args.state_dict)
    n_real = sum(t.raw.size for t in tensors)
    print(f"[lossy-coarsen] {len(tensors)} tensors, total {n_real:,} symbols")
    print(f"[lossy-coarsen] PR101 brotli baseline: {PR101_BROTLI_BASELINE_BYTES:,} B")

    all_results: dict = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[MPS-research-signal]",
        "evidence_semantics": "mps_or_cpu_byte_roundtrip_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "family_falsified": False,
        "falsification_scope": "none_proxy_anchor_only",
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "input_state_dict": str(args.state_dict),
        "n_real_symbols": n_real,
        "uniform_K_baselines": [],
        "per_tensor_K_results": [],
    }

    # Uniform-K baselines (single K applied to all tensors)
    print()
    print("--- uniform-K baselines (single K everywhere) ---")
    print(f"{'K':>4} {'rel_err':>9} {'payload_B':>11} {'archive_B':>11} {'delta_baseline':>14}")
    uniform_Ks = [int(x) for x in args.uniform_Ks.split(",") if x.strip()]
    for K in uniform_Ks:
        meas = encode_with_uniform_K(tensors, K, brotli_quality=args.brotli_quality)
        delta = meas["archive_bytes"] - PR101_BROTLI_BASELINE_BYTES
        print(
            f"{K:>4} {meas['rel_err']:>9.4f} {meas['payload_brotli_bytes']:>11,} "
            f"{meas['archive_bytes']:>11,} {delta:>+14,}"
        )
        all_results["uniform_K_baselines"].append({
            "K": K,
            "rel_err": meas["rel_err"],
            "payload_brotli_bytes": meas["payload_brotli_bytes"],
            "archive_bytes": meas["archive_bytes"],
            "delta_baseline": delta,
        })

    # Per-tensor K (each tensor uses its own K, chosen for given budget)
    print()
    print("--- per-tensor K (each tensor has its own K within rel_err budget) ---")
    print(f"{'budget':>9} {'rel_err':>9} {'payload_B':>11} {'archive_B':>11} {'delta_baseline':>14}")
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    for budget in budgets:
        Ks = []
        for tb in tensors:
            K, _ = find_best_K_for_tensor(tb.raw, budget=budget)
            Ks.append(K)
        meas = encode_with_per_tensor_K(tensors, Ks, brotli_quality=args.brotli_quality)
        delta = meas["archive_bytes"] - PR101_BROTLI_BASELINE_BYTES
        print(
            f"{budget:>9.4f} {meas['rel_err']:>9.4f} {meas['payload_brotli_bytes']:>11,} "
            f"{meas['archive_bytes']:>11,} {delta:>+14,}"
        )
        all_results["per_tensor_K_results"].append({
            "budget": budget,
            "rel_err": meas["rel_err"],
            "payload_brotli_bytes": meas["payload_brotli_bytes"],
            "archive_bytes": meas["archive_bytes"],
            "delta_baseline": delta,
            "Ks": Ks,
        })

    # Find best run that beats baseline AND has rel_err < 0.05
    candidates = []
    for r in all_results["uniform_K_baselines"]:
        if r["rel_err"] < 0.05 and r["delta_baseline"] < 0:
            candidates.append(("uniform_K", r))
    for r in all_results["per_tensor_K_results"]:
        if r["rel_err"] < 0.05 and r["delta_baseline"] < 0:
            candidates.append(("per_tensor_K", r))
    if candidates:
        # Prefer smallest archive_bytes (winning the byte race)
        best_kind, best = min(candidates, key=lambda x: x[1]["archive_bytes"])
    else:
        best_kind, best = None, None

    print()
    print("=" * 70)
    print("[lossy-coarsen] SUMMARY")
    print("=" * 70)
    if best is not None:
        print("BEST (rel_err < 0.05 AND byte-lower than baseline):")
        print(f"  kind          : {best_kind}")
        print(f"  config        : {('budget=' + str(best['budget'])) if 'budget' in best else ('K=' + str(best['K']))}")
        print(f"  archive_bytes : {best['archive_bytes']:,} B  [MPS-research-signal]")
        print(f"  vs baseline   : {best['delta_baseline']:+,} B")
        print(f"  rel_err       : {best['rel_err']:.4f}")
    else:
        print("NO config met both (rel_err < 0.05 AND archive < 178,144 B)")

    if best is not None:
        all_results["best_kind"] = best_kind
        all_results["best_archive_bytes"] = best["archive_bytes"]
        all_results["best_rel_err"] = best["rel_err"]
        if "K" in best:
            all_results["best_config"] = f"uniform_K={best['K']}"
        else:
            all_results["best_config"] = f"per_tensor_K_budget={best['budget']}"
    (args.output_dir / "manifest.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if best is not None:
        verdict = (
            f"BYTE-LOWER-THAN-PR101-brotli ({best['delta_baseline']:+,} B) "
            f"at proxy rel_err {best['rel_err']:.4f}; "
            "requires byte-closed runtime packet and exact CUDA auth eval"
        )
        evidence_row = {
            "technique": "lossy_coarsening_analytical",
            "empirical_archive_bytes": best["archive_bytes"],
            "empirical_rel_err": best["rel_err"],
            "evidence_grade": "[MPS-research-signal]",
            "evidence_semantics": all_results["evidence_semantics"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "source": (
                f"[MPS-research-signal] {args.output_dir}/manifest.json "
                f"(analytical per-tensor K coarsening; prior CompressAI smoke "
                f"configs did not reach rel_err < 0.05 on PR101 substrate; "
                f"not a neural-codec family kill)"
            ),
            "timestamp": timestamp,
            "contest_dispatch_verdict": verdict,
            "supersedes_prior_DEFERRED_audit": True,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "family_falsified": False,
            "falsification_scope": "none_proxy_anchor_only",
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
        }
        args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"[lossy-coarsen] evidence row appended to {args.evidence_jsonl}")
    else:
        verdict = "DEFERRED-pending-research (no config met both constraints)"
    print(f"verdict  : {verdict}")
    print(f"manifest : {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
