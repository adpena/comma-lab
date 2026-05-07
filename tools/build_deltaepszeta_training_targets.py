"""Build a per-tensor δεζ training-target table from Path B Shannon analysis.

Per Grand Council 2026-05-07 verdict, the strategic follow-on after
PR103-on-PR106 standalone (-661 B) is **joint substrate training to reduce
native description length**. The training objective is **H₂ → H₀ collapse**:
make weights conditionally uniform so context-aware coding hits the i.i.d.
Shannon floor.

This tool consumes the per-tensor Shannon analysis JSON (Path B output) and
produces a δεζ training-target table. The table prescribes:

- per-tensor `headroom_bits` = max(0, H₀ - H₂)  — how much further the tensor
  could be compressed by context-aware coding if H₂ collapsed to H₀.
  (Note: H₂ ≤ H₀, so headroom_bits ≥ 0 always; the per-tensor JSON values are
  pre-computed and obey this invariant.)
- per-tensor `loss_weight` ∝ headroom_bits × n_symbols  — total bytes saveable.
  Tensors with the largest aggregate headroom get the largest training-loss
  weight, focusing optimizer attention where the rate prize is biggest.
- aggregate stats: total bytes saveable, top-K tensors by headroom, weights
  normalized to sum to 1.

The table is consumed by `tac.codec_pipeline_deltaepszeta_callback` (when
δεζ training lanes land) AND by `tac.shannon_h2_loss.shannon_h2_loss` as
per-tensor multipliers in the auxiliary loss term.

Usage:
    python tools/build_deltaepszeta_training_targets.py \\
        --shannon-json experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json \\
        --output experiments/results/lane_deltaepszeta_targets_pr106_<UTC>/targets.json

Output: ``targets.json`` with per-tensor weights + a markdown table for the writeup.

Strict-scorer-rule: pure CPU + numpy + json. No torch, no MPS, no scorer load.
All scores tagged ``[empirical:<source path>]`` per CLAUDE.md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import pathlib
import sys
from typing import Any


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_shannon_json(pattern: str) -> pathlib.Path:
    """Resolve a glob to a single Shannon analysis JSON; pick newest."""
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"no Shannon analysis JSON matched: {pattern}; "
            f"run tools/per_tensor_shannon_analysis.py first"
        )
    return pathlib.Path(candidates[-1])


def build_targets(
    shannon_json_path: pathlib.Path,
) -> dict[str, Any]:
    """Read Path B Shannon analysis JSON and produce a δεζ target table."""
    raw = json.loads(shannon_json_path.read_text())
    per_tensor = raw["per_tensor"]
    if len(per_tensor) != raw.get("n_tensors", len(per_tensor)):
        raise ValueError(
            f"per_tensor length {len(per_tensor)} != n_tensors "
            f"{raw.get('n_tensors')}"
        )

    rows: list[dict[str, Any]] = []
    for t in per_tensor:
        h0 = float(t["H0_bits"])
        h2 = float(t["H2_bits"])
        n_symbols = int(t["n_symbols"])
        headroom_bits = max(0.0, h0 - h2)
        # Saveable bytes if H₂ collapses to its current H₂ value (no further
        # savings available); the headroom is the gap to *current* H₀ which
        # represents the i.i.d.-floor we'd hit if a perfect context-aware coder
        # exhausted the conditional structure. Different framing: the rate
        # PRIZE if context-aware coding fully exploits H₂ vs current brotli.
        # We use H₀ - H₂ as the conservative "δεζ training prize":
        # if training drives the weights toward H₂ ≈ H₀ (uniform conditional),
        # AC and brotli equalize; if training preserves H₂ < H₀, context-aware
        # coding wins by `headroom_bits × n_symbols / 8` bytes.
        prize_bytes = headroom_bits * n_symbols / 8.0
        rows.append({
            "idx": int(t["idx"]),
            "name": str(t["name"]),
            "n_symbols": n_symbols,
            "in_pr103_ac_set": bool(t.get("in_pr103_ac_set", False)),
            "H0_bits": h0,
            "H1_bits": float(t["H1_bits"]),
            "H2_bits": h2,
            "headroom_bits": headroom_bits,
            "prize_bytes": prize_bytes,
            "brotli_bytes": int(t["brotli_bytes"]),
            "ac_bytes": int(t["ac_bytes"]) if t.get("ac_bytes") is not None else None,
        })

    total_prize = sum(r["prize_bytes"] for r in rows)
    total_n = sum(r["n_symbols"] for r in rows)
    # Normalize per-tensor loss weights to sum to 1.
    if total_prize > 0:
        for r in rows:
            r["loss_weight_normalized"] = r["prize_bytes"] / total_prize
    else:
        for r in rows:
            r["loss_weight_normalized"] = 0.0

    rows_sorted = sorted(rows, key=lambda r: r["prize_bytes"], reverse=True)

    return {
        "started_at_utc": _utc_iso(),
        "tool": "tools/build_deltaepszeta_training_targets",
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "[empirical]",
        "source_shannon_analysis": str(shannon_json_path),
        "substrate": raw.get("substrate"),
        "n_tensors": len(rows),
        "summary": {
            "total_prize_bytes": math.ceil(total_prize),
            "total_n_symbols": total_n,
            "total_brotli_bytes": int(raw["summary"]["total_brotli_bytes"]),
            "total_shannon_floor_h0_bytes": int(
                raw["summary"]["total_shannon_floor_h0_bytes"]
            ),
            "total_shannon_floor_h2_bytes": int(
                raw["summary"]["total_shannon_floor_h2_bytes"]
            ),
            "ratio_h2_over_h0": (
                raw["summary"]["total_shannon_floor_h2_bytes"]
                / max(1, raw["summary"]["total_shannon_floor_h0_bytes"])
            ),
        },
        "per_tensor": rows_sorted,
        "top10_by_prize_bytes": [
            {
                "idx": r["idx"],
                "name": r["name"],
                "prize_bytes": math.ceil(r["prize_bytes"]),
                "headroom_bits": round(r["headroom_bits"], 4),
                "loss_weight": round(r["loss_weight_normalized"], 4),
            }
            for r in rows_sorted[:10]
        ],
    }


def render_markdown(targets: dict[str, Any]) -> str:
    """Render a markdown summary suitable for paper / writeup inclusion."""
    s = targets["summary"]
    lines = [
        "# δεζ Training Targets — PR106 substrate",
        "",
        f"**Source**: `{targets['source_shannon_analysis']}` `[empirical]`",
        f"**Substrate**: `{targets['substrate']}`",
        f"**N tensors**: {targets['n_tensors']}",
        f"**Total prize bytes (if H₂→H₀ collapses fully)**: {s['total_prize_bytes']:,} B",
        f"**Total brotli bytes today**: {s['total_brotli_bytes']:,} B",
        f"**H₂/H₀ ratio aggregate**: {s['ratio_h2_over_h0']:.4f}",
        "",
        "## Top 10 tensors by training-prize bytes",
        "",
        "| rank | idx | name | n_symbols | H₀ | H₂ | headroom (bits) | prize (B) | weight |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(targets["per_tensor"][:10], 1):
        lines.append(
            f"| {rank} | {r['idx']} | `{r['name']}` | {r['n_symbols']:,} | "
            f"{r['H0_bits']:.4f} | {r['H2_bits']:.4f} | "
            f"{r['headroom_bits']:.4f} | {math.ceil(r['prize_bytes']):,} | "
            f"{r['loss_weight_normalized']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The `headroom_bits` column = H₀ - H₂; this is the per-symbol bit",
        "  prize if a context-aware coder fully exploited H₂'s conditional",
        "  structure (vs current brotli sitting at ~1.015× H₀).",
        "- The `prize (B)` column = headroom_bits × n_symbols / 8; this is",
        "  the per-tensor byte budget that a δεζ-trained substrate could",
        "  unlock.",
        "- The `weight` column normalizes prizes to sum to 1; use as a",
        "  per-tensor multiplier in the δεζ auxiliary loss term.",
        "",
        "## How to apply",
        "",
        "```python",
        "import torch",
        "from tac.shannon_h2_loss import shannon_h2_loss",
        "import json",
        "",
        "targets = json.load(open('targets.json'))",
        "weights = {r['name']: r['loss_weight_normalized'] for r in targets['per_tensor']}",
        "",
        "def deltaepszeta_aux_loss(state_dict):",
        "    total = torch.tensor(0.0)",
        "    for name, w_value in state_dict.items():",
        "        if name in weights:",
        "            h2 = shannon_h2_loss(w_value, n_bits=8)",
        "            total = total + weights[name] * h2",
        "    return total",
        "```",
        "",
        "Score claims: **none**. This is a training-target derivation; final",
        "score evidence comes from contest-CUDA `archive.zip → inflate.sh →",
        "upstream/evaluate.py` on the produced archive bytes.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build δεζ training-target table from Shannon analysis"
    )
    p.add_argument(
        "--shannon-json",
        default="experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json",
        help="path or glob to Shannon analysis JSON",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="output dir (default: experiments/results/lane_deltaepszeta_targets_pr106_<UTC>/)",
    )
    args = p.parse_args(argv)

    shannon_path = _resolve_shannon_json(args.shannon_json)
    targets = build_targets(shannon_path)

    if args.output_dir is None:
        ts = _utc_timestamp()
        out_dir = pathlib.Path(f"experiments/results/lane_deltaepszeta_targets_pr106_{ts}")
    else:
        out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "targets.json"
    md_path = out_dir / "targets.md"
    json_path.write_text(json.dumps(targets, indent=2))
    md_path.write_text(render_markdown(targets))

    print(f"shannon_source: {shannon_path}")
    print(f"output_dir:     {out_dir}")
    print(f"  targets.json: {json_path.stat().st_size:,} B")
    print(f"  targets.md:   {md_path.stat().st_size:,} B")
    print()
    s = targets["summary"]
    print(f"total_prize_bytes (H₂→H₀ if exhausted): {s['total_prize_bytes']:,}")
    print(f"total_brotli_bytes today:               {s['total_brotli_bytes']:,}")
    print(f"H₂/H₀ aggregate ratio:                  {s['ratio_h2_over_h0']:.4f}")
    print()
    print("top 5 tensors by prize:")
    for r in targets["per_tensor"][:5]:
        print(
            f"  idx={r['idx']:>2} {r['name']:<24s} "
            f"prize={math.ceil(r['prize_bytes']):>6,} B  "
            f"weight={r['loss_weight_normalized']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
