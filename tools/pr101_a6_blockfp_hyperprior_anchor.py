#!/usr/bin/env python3
"""PR101 A6 byte-anchor — Selfcomp block-FP × Ballé hyperprior compose.

Council R6 (Selfcomp + Ballé seats). The compose hypothesis: per-block scale
parameter (Selfcomp's contribution) is exactly the conditioning sigma the
hyperprior wants (Ballé's contribution); together they should produce
tighter entropy estimates than either standalone.

This anchor sweeps ``block_size ∈ {16, 32, 64, 128} × scale_quant ∈
{fp16, uint8}`` over PR101's INT8 symbol stream (concat across all 28
tensors in ``FIXED_STATE_SCHEMA``) and reports byte savings vs three
baselines:

1. PR101 brotli baseline: 178,144 B (canonical, see
   ``tools/pr101_lossy_coarsening_analytical.py``).
2. Block-FP standalone: per-block scales + raw int8 residuals.
3. Hyperprior standalone: global Gaussian σ over the whole stream.

Each ablation cell is byte-anchored and tagged
``[byte-anchor; codec=a6_selfcomp_blockfp_hyperprior_compose]`` per the
A2 packet-clearance evidence-kind separation rule (Catalog #115). The
compose is LOSSLESS on the int8 stream — roundtrip is byte-identical
(verified by every unit test in ``test_a6_blockfp_hyperprior_compose_unit.py``)
— so this is a pure byte-anchor lane: any score claim requires an
exact CUDA auth eval on a byte-closed runtime packet.

Output:
* ``manifest.json`` with the full sweep and one ``best`` row.
* ``build_manifest.json`` with the dispatch-eligibility ledger
  (``score_claim=False``, ``byte_proxy_only=True``,
  ``ready_for_exact_eval_dispatch=False``,
  ``dispatch_blockers=[awaiting_compose_vs_baseline_dispatch_comparison, ...]``).
* one row appended to ``reports/cathedral_autopilot_evidence.jsonl``.

Usage:
    python tools/pr101_a6_blockfp_hyperprior_anchor.py \\
        --state-dict experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.a6_selfcomp_blockfp_hyperprior_compose import (  # noqa: E402
    SCALE_QUANT_FP16,
    SCALE_QUANT_UINT8,
    compose_blockfp_with_hyperprior,
    encode_blockfp_only,
    encode_hyperprior_only,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)


TOOL_NAME = "tools/pr101_a6_blockfp_hyperprior_anchor.py"
SCHEMA_VERSION = "pr101_a6_blockfp_hyperprior_anchor.v1"
CODEC_TAG = "a6_selfcomp_blockfp_hyperprior_compose"

# Canonical PR101 byte constants (cross-ref ``tools/pr101_lossy_coarsening_analytical.py``).
ARCHIVE_OVERHEAD_BYTES = 16_094
PR101_BROTLI_BASELINE_BYTES = 178_144
N_TENSORS = len(FIXED_STATE_SCHEMA)

# Council-aligned ablation cells.
DEFAULT_BLOCK_SIZES = (16, 32, 64, 128)
DEFAULT_SCALE_QUANTS = (
    ("fp16", SCALE_QUANT_FP16),
    ("uint8", SCALE_QUANT_UINT8),
)


DISPATCH_BLOCKERS = [
    "measured_config_above_pr101_brotli_current_proxy",
    "wire_format_proxy_not_selfcomp_per_channel_block_fp",
    "awaiting_compose_vs_baseline_dispatch_comparison",
    "missing_exact_cuda_auth_eval",
    "no_runtime_dequantize_path_built",
    "byte_closed_a6_runtime_packet_missing",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "proxy_byte_anchor_not_score_evidence",
]


def collect_pr101_int8_stream(state_dict_path: Path) -> tuple[np.ndarray, int]:
    """Concat the int8 symbol stream from every PR101 tensor.

    Returns ``(stream, n_real_symbols)`` where ``stream`` is 1-D int8.
    """
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    chunks: list[np.ndarray] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        chunks.append(qt.q_i8.astype(np.int8).reshape(-1))
    flat = np.concatenate(chunks)
    return flat, int(flat.size)


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
        "--block-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BLOCK_SIZES),
        help="Comma-separated block-size sweep (e.g. '16,32,64,128').",
    )
    p.add_argument(
        "--scale-quants",
        type=str,
        default="fp16,uint8",
        help="Comma-separated scale-quant modes from {fp16, uint8, fp32}.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--evidence-jsonl",
        type=Path,
        default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl",
    )
    p.add_argument(
        "--no-evidence-append",
        action="store_true",
        help="Skip the cathedral_autopilot evidence append (dry-run mode).",
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")
    if args.output_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = (
            REPO_ROOT / f"experiments/results/pr101_a6_blockfp_hyperprior_{ts}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[a6-anchor] state_dict      : {args.state_dict}")
    print(f"[a6-anchor] output          : {args.output_dir}")
    print(f"[a6-anchor] codec_tag       : {CODEC_TAG}")

    stream, n_real = collect_pr101_int8_stream(args.state_dict)
    print(f"[a6-anchor] n_real_symbols  : {n_real:,}")
    print(f"[a6-anchor] stream dtype    : {stream.dtype}")
    print(f"[a6-anchor] stream range    : [{int(stream.min())}, {int(stream.max())}]")
    print(f"[a6-anchor] PR101 baseline  : {PR101_BROTLI_BASELINE_BYTES:,} B (brotli)")

    # ── Standalone baselines ──────────────────────────────────────────────
    print()
    print("--- standalone baselines (block-FP only / hyperprior only) ---")
    print(f"{'baseline':>22} {'payload_B':>11} {'side_B':>9} {'archive_B':>11} {'delta_PR101':>13}")

    blockfp_only_rows: list[dict] = []
    for bsz in [int(x) for x in args.block_sizes.split(",") if x.strip()]:
        _, ledger = encode_blockfp_only(stream, block_size=bsz, scale_quant=SCALE_QUANT_FP16)
        archive_b = ledger["total_bytes"] + ARCHIVE_OVERHEAD_BYTES
        delta = archive_b - PR101_BROTLI_BASELINE_BYTES
        print(
            f"{'blockfp_only B=' + str(bsz):>22} "
            f"{ledger['payload_bytes']:>11,} "
            f"{ledger['side_info_bytes']:>9,} "
            f"{archive_b:>11,} "
            f"{delta:>+13,}"
        )
        blockfp_only_rows.append({
            "block_size": bsz,
            "scale_quant": "fp16",
            "payload_bytes": ledger["payload_bytes"],
            "side_info_bytes": ledger["side_info_bytes"],
            "compose_bytes": ledger["total_bytes"],
            "archive_bytes": archive_b,
            "delta_baseline": delta,
        })

    _, ledger_h = encode_hyperprior_only(stream)
    archive_h = ledger_h["total_bytes"] + ARCHIVE_OVERHEAD_BYTES
    delta_h = archive_h - PR101_BROTLI_BASELINE_BYTES
    print(
        f"{'hyperprior_only_global':>22} "
        f"{ledger_h['payload_bytes']:>11,} "
        f"{ledger_h['side_info_bytes']:>9,} "
        f"{archive_h:>11,} "
        f"{delta_h:>+13,}"
    )
    hyperprior_only_row = {
        "encoding": "hyperprior_only_global_sigma",
        "payload_bytes": ledger_h["payload_bytes"],
        "side_info_bytes": ledger_h["side_info_bytes"],
        "compose_bytes": ledger_h["total_bytes"],
        "archive_bytes": archive_h,
        "delta_baseline": delta_h,
    }

    # ── Compose ablation cells ───────────────────────────────────────────
    print()
    print("--- compose: blockfp × hyperprior (ablation cells) ---")
    print(f"{'B':>4} {'sq':>5} {'payload_B':>11} {'side_B':>9} {'compose_B':>11} {'archive_B':>11} {'delta_PR101':>13}")

    SCALE_QUANT_MAP = dict(DEFAULT_SCALE_QUANTS)
    SCALE_QUANT_MAP["fp32"] = 1  # SCALE_QUANT_FP32

    block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    scale_quant_names = [x.strip() for x in args.scale_quants.split(",") if x.strip()]

    compose_rows: list[dict] = []
    for bsz in block_sizes:
        for sq_name in scale_quant_names:
            sq_code = SCALE_QUANT_MAP[sq_name]
            _, ledger = compose_blockfp_with_hyperprior(
                stream,
                block_size=bsz,
                scale_quant=sq_code,
                verify_roundtrip=True,
            )
            archive_b = ledger["total_bytes"] + ARCHIVE_OVERHEAD_BYTES
            delta = archive_b - PR101_BROTLI_BASELINE_BYTES
            print(
                f"{bsz:>4} {sq_name:>5} "
                f"{ledger['payload_bytes']:>11,} "
                f"{ledger['side_info_bytes']:>9,} "
                f"{ledger['total_bytes']:>11,} "
                f"{archive_b:>11,} "
                f"{delta:>+13,}"
            )
            compose_rows.append({
                "block_size": bsz,
                "scale_quant": sq_name,
                "n_blocks": ledger["n_blocks"],
                "payload_bytes": ledger["payload_bytes"],
                "side_info_bytes": ledger["side_info_bytes"],
                "header_bytes": ledger["header_bytes"],
                "compose_bytes": ledger["total_bytes"],
                "archive_bytes": archive_b,
                "delta_baseline": delta,
            })

    # ── Identify the best compose cell ───────────────────────────────────
    if compose_rows:
        best = min(compose_rows, key=lambda r: r["archive_bytes"])
    else:
        best = None
    best_blockfp_only = (
        min(blockfp_only_rows, key=lambda r: r["archive_bytes"])
        if blockfp_only_rows else None
    )

    # Compose vs standalone baseline comparisons
    if best is not None:
        compose_vs_blockfp = (
            best["archive_bytes"] - best_blockfp_only["archive_bytes"]
            if best_blockfp_only else None
        )
        compose_vs_hyperprior = best["archive_bytes"] - hyperprior_only_row["archive_bytes"]
        measured_config_negative = best["delta_baseline"] >= 0
    else:
        compose_vs_blockfp = compose_vs_hyperprior = None
        measured_config_negative = False

    # ── manifest.json (research/proxy artifact) ──────────────────────────
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "codec_tag": CODEC_TAG,
        "timestamp": timestamp,
        "evidence_grade": "[byte-anchor]",
        "evidence_semantics": "byte_roundtrip_proxy_no_score",
        "score_claim": False,
        "byte_proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "cuda_eval_worth_testing": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "family_falsified": False,
        "measured_config_negative": measured_config_negative,
        "falsification_scope": (
            "current_max_abs_scale_conditional_range_coder_proxy_only"
            if measured_config_negative
            else "none_proxy_anchor_only"
        ),
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "input_state_dict": str(args.state_dict),
        "n_real_symbols": n_real,
        "pr101_brotli_baseline_bytes": PR101_BROTLI_BASELINE_BYTES,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "blockfp_only_baselines": blockfp_only_rows,
        "hyperprior_only_baseline": hyperprior_only_row,
        "compose_cells": compose_rows,
        "best_compose": best,
        "best_blockfp_only": best_blockfp_only,
        "compose_vs_blockfp_only_delta": compose_vs_blockfp,
        "compose_vs_hyperprior_only_delta": compose_vs_hyperprior,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # ── build_manifest.json (dispatch-eligibility ledger) ────────────────
    build_manifest = {
        "schema": "pr101_a6_blockfp_hyperprior.build_manifest.v1",
        "tool": TOOL_NAME,
        "codec_tag": CODEC_TAG,
        "timestamp": timestamp,
        "score_claim": False,
        "byte_proxy_only": True,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "measured_config_negative": measured_config_negative,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "cleared_blockers": [],
        "cleared_blockers_by_evidence": {},
        "next_required_actions": [
            "wire byte-closed runtime packet (inflate.sh + decompose path) consuming the compose codec",
            "run inflate parity smoke against PR101 baseline archive",
            "dispatch exact CUDA auth eval on the byte-closed runtime packet",
        ],
        "best_archive_bytes": best["archive_bytes"] if best else None,
        "best_block_size": best["block_size"] if best else None,
        "best_scale_quant": best["scale_quant"] if best else None,
        "compose_vs_blockfp_only_delta": compose_vs_blockfp,
        "compose_vs_hyperprior_only_delta": compose_vs_hyperprior,
        "delta_pr101_brotli_baseline": (
            best["delta_baseline"] if best else None
        ),
    }
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2), encoding="utf-8"
    )

    # ── Cathedral autopilot evidence row ─────────────────────────────────
    if best is not None:
        verdict_word = (
            "byte_proxy_BEATS_pr101_brotli_requires_runtime_packet"
            if best["delta_baseline"] < 0
            else "measured_config_negative_above_pr101_brotli"
        )
        if compose_vs_blockfp is not None and compose_vs_blockfp < 0:
            verdict_word += "_AND_BEATS_blockfp_only"
        if compose_vs_hyperprior < 0:
            verdict_word += "_AND_BEATS_hyperprior_only"

        verdict_text = (
            f"{verdict_word}: best={best['archive_bytes']:,} B "
            f"(B={best['block_size']}, sq={best['scale_quant']}) "
            f"vs PR101 brotli {PR101_BROTLI_BASELINE_BYTES:,} B "
            f"({best['delta_baseline']:+,} B); "
            "requires byte-closed runtime packet and exact CUDA auth eval"
        )

        evidence_row = {
            "technique": "a6_selfcomp_blockfp_hyperprior_compose",
            "codec_tag": CODEC_TAG,
            "empirical_archive_bytes": best["archive_bytes"],
            "evidence_grade": "[byte-anchor]",
            "evidence_semantics": "byte_roundtrip_proxy_no_score",
            "score_claim": False,
            "byte_proxy_only": True,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "cuda_eval_worth_testing": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "source": (
                f"[byte-anchor; codec={CODEC_TAG}] "
                f"{args.output_dir}/manifest.json "
                "(Selfcomp block-FP × Ballé hyperprior compose; "
                "lossless on int8 stream; per-block scale = hyperprior σ; "
                "no scorer load; no neural-net weights shipped)"
            ),
            "timestamp": timestamp,
            "contest_dispatch_verdict": verdict_text,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "downstream_selection_can_change_charged_bits": True,
            "family_falsified": False,
            "measured_config_negative": measured_config_negative,
            "falsification_scope": (
                "current_max_abs_scale_conditional_range_coder_proxy_only"
                if measured_config_negative
                else "none_proxy_anchor_only"
            ),
            "dispatch_blockers": list(DISPATCH_BLOCKERS),
            "compose_vs_blockfp_only_delta": compose_vs_blockfp,
            "compose_vs_hyperprior_only_delta": compose_vs_hyperprior,
            "best_block_size": best["block_size"],
            "best_scale_quant": best["scale_quant"],
        }
        if not args.no_evidence_append:
            args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.evidence_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evidence_row) + "\n")
            print(f"[a6-anchor] evidence row appended to {args.evidence_jsonl}")
    else:
        verdict_text = "DEFERRED-pending-research (no compose cells produced output)"

    # ── Final summary ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("[a6-anchor] SUMMARY")
    print("=" * 78)
    if best is not None:
        print("BEST compose cell:")
        print(
            f"  block_size      : {best['block_size']}  scale_quant: {best['scale_quant']}"
        )
        print(f"  archive_bytes   : {best['archive_bytes']:,} B  [byte-anchor]")
        print(
            f"  vs PR101 brotli : {best['delta_baseline']:+,} B "
            f"(baseline {PR101_BROTLI_BASELINE_BYTES:,} B)"
        )
        if compose_vs_blockfp is not None:
            print(
                f"  vs blockfp-only : {compose_vs_blockfp:+,} B "
                f"(baseline {best_blockfp_only['archive_bytes']:,} B)"
            )
        print(
            f"  vs hyperprior-only: {compose_vs_hyperprior:+,} B "
            f"(baseline {hyperprior_only_row['archive_bytes']:,} B)"
        )
    else:
        print("NO compose cells produced output")
    print(f"verdict  : {verdict_text}")
    print(f"manifest : {args.output_dir / 'manifest.json'}")
    print(f"build_mf : {args.output_dir / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
