# SPDX-License-Identifier: MIT
"""Build per-tensor delta-epsilon-zeta training targets from Shannon analysis.

Per Grand Council 2026-05-07 verdict, the strategic follow-on after
PR103-on-PR106 standalone (-661 B) is **joint substrate training to reduce
native description length**. This tool does not claim that forcing H2 toward H0
creates savings. It measures the current H0-H2 conditional-entropy gap and uses
that gap to prioritize tensors for context-aware coding or substrate-training
experiments.

This tool consumes the per-tensor Shannon analysis JSON (Path B output) and
produces a delta-epsilon-zeta training-target table. The table prescribes:

- per-tensor `headroom_bits` = max(0, H0 - H2) -- how much current
  conditional structure a context-aware coder could exploit.
  (Note: H2 <= H0, so headroom_bits >= 0 always; the per-tensor JSON values are
  pre-computed and obey this invariant.)
- per-tensor `loss_weight` proportional to headroom_bits * n_symbols -- total
  current conditional-entropy gap in bytes.
  Tensors with the largest aggregate headroom get the largest training-loss
  weight, focusing optimizer attention where the rate prize is biggest.
- aggregate stats: total bytes saveable, top-K tensors by headroom, weights
  normalized to sum to 1.

The table is consumed by `tac.codec_pipeline_deltaepszeta_callback` (when
delta-epsilon-zeta training lanes land) and by
`tac.shannon_h2_loss.shannon_h2_loss` as per-tensor multipliers in auxiliary
loss terms.

Usage:
    python tools/build_deltaepszeta_training_targets.py \\
        --shannon-json experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json \\
        --output-dir experiments/results/lane_deltaepszeta_targets_pr106_<UTC>/

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
from collections.abc import Sequence
from typing import Any

ENTROPY_ORDER_TOLERANCE_BITS = 1e-9


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_shannon_json(patterns: str | Sequence[str]) -> pathlib.Path:
    """Resolve one or more path/glob operands to a single Shannon JSON.

    Bug-hunter v2 fix 2026-05-07 (MEDIUM): the previous version sorted
    candidates lexicographically and picked the lex-last entry. That works
    when filenames carry a UTC timestamp in a strictly-increasing format
    (e.g. ``..._20260507T184921Z.json``) but fails subtly when:

    1. A file is renamed to ``..._LATEST.json`` (lex-late but stale, or
       lex-early-but-latest depending on sibling names).
    2. Two timestamp formats coexist (e.g. ``2026-05-07T18`` vs
       ``20260507T180000Z``) so lex-order disagrees with mtime-order.
    3. A backup or hand-edit produces a file with a lex-late name but an
       earlier mtime.

    Now we sort by ``(mtime, lex)`` so the most-recently-modified file wins,
    with lex-order as a deterministic tiebreaker for builds that produce
    multiple files in the same second. This matches the operator intent
    ("pick the newest analysis") under both naming conventions.

    Bug-hunter v3 fix 2026-05-09 (LOW): the CLI help advertises
    ``--shannon-json`` as "path or glob", but in zsh an unquoted glob can
    expand into multiple argv operands before Python sees it. Accepting a
    sequence here keeps quoted globs, exact paths, and shell-expanded globs
    equivalent.
    """
    if isinstance(patterns, str):
        operands = [patterns]
    else:
        operands = list(patterns)
    candidates: list[str] = []
    for operand in operands:
        matches = glob.glob(operand)
        if matches:
            candidates.extend(matches)
        elif pathlib.Path(operand).exists():
            candidates.append(operand)
    if not candidates:
        raise FileNotFoundError(
            f"no Shannon analysis JSON matched: {operands}; "
            f"run tools/per_tensor_shannon_analysis.py first"
        )
    candidates_sorted = sorted(
        candidates,
        key=lambda p: (pathlib.Path(p).stat().st_mtime, p),
    )
    return pathlib.Path(candidates_sorted[-1])


def build_targets(
    shannon_json_path: pathlib.Path,
    *,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    """Read Path B Shannon analysis JSON and produce a target table."""
    raw = json.loads(shannon_json_path.read_text(encoding="utf-8"))
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
        if h2 > h0 + ENTROPY_ORDER_TOLERANCE_BITS:
            raise ValueError(
                "entropy order invariant violated for tensor "
                f"{t.get('idx')!r} {t.get('name')!r}: H2_bits={h2} > H0_bits={h0}"
            )
        headroom_bits = max(0.0, h0 - h2)
        # Current conditional-entropy gap. If a context-aware coder can exploit
        # the measured H2 structure, this is the per-tensor rate prize versus
        # an i.i.d. H0 code. A training objective must use this as a weighting
        # signal, not as a proof that driving H2 toward H0 creates savings.
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
        "started_at_utc": started_at_utc or _utc_iso(),
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
        "# Delta-Epsilon-Zeta Training Targets - PR106 substrate",
        "",
        f"**Source**: `{targets['source_shannon_analysis']}` `[empirical]`",
        f"**Substrate**: `{targets['substrate']}`",
        f"**N tensors**: {targets['n_tensors']}",
        f"**Total current H0-H2 gap**: {s['total_prize_bytes']:,} B",
        f"**Total brotli bytes today**: {s['total_brotli_bytes']:,} B",
        f"**H2/H0 ratio aggregate**: {s['ratio_h2_over_h0']:.4f}",
        "",
        "## Top 10 tensors by training-prize bytes",
        "",
        "| rank | idx | name | n_symbols | H0 | H2 | headroom (bits) | prize (B) | weight |",
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
        "- The `headroom_bits` column = H0 - H2; this is the current",
        "  per-symbol conditional-entropy gap a context-aware coder could",
        "  exploit (vs current brotli sitting at about 1.015x H0).",
        "- The `prize (B)` column = headroom_bits * n_symbols / 8; this is",
        "  a per-tensor byte-gap weighting signal for delta-epsilon-zeta",
        "  substrate experiments.",
        "- The `weight` column normalizes prizes to sum to 1; use as a",
        "  per-tensor multiplier in the auxiliary loss term.",
        "",
        "## How to apply",
        "",
        "The example below is a **structured-H2 experiment**: it minimizes H2",
        "on tensors with the largest current H0-H2 gap. It is only useful when",
        "the produced archive uses a context-aware coder that can exploit that",
        "same conditional structure. It is not a direct proof of score or byte",
        "savings.",
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
        "score evidence comes from contest-CUDA `archive.zip -> inflate.sh ->",
        "upstream/evaluate.py` on the produced archive bytes.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build delta-epsilon-zeta training-target table from Shannon analysis"
    )
    p.add_argument(
        "--shannon-json",
        default="experiments/results/lane_per_tensor_shannon_pr106_*/per_tensor_shannon.json",
        nargs="+",
        help="path or glob to Shannon analysis JSON",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="output dir (default: experiments/results/lane_deltaepszeta_targets_pr106_<UTC>/)",
    )
    p.add_argument(
        "--started-at-utc",
        default=None,
        help="fixed ISO-8601 UTC timestamp for byte-reproducible output",
    )
    args = p.parse_args(argv)

    shannon_path = _resolve_shannon_json(args.shannon_json)
    targets = build_targets(shannon_path, started_at_utc=args.started_at_utc)

    if args.output_dir is None:
        ts = _utc_timestamp()
        out_dir = pathlib.Path(f"experiments/results/lane_deltaepszeta_targets_pr106_{ts}")
    else:
        out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "targets.json"
    md_path = out_dir / "targets.md"
    json_path.write_text(
        json.dumps(targets, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(targets) + "\n", encoding="utf-8")

    print(f"shannon_source: {shannon_path}")
    print(f"output_dir:     {out_dir}")
    print(f"  targets.json: {json_path.stat().st_size:,} B")
    print(f"  targets.md:   {md_path.stat().st_size:,} B")
    print()
    s = targets["summary"]
    print(f"total_h0_h2_gap_bytes:                  {s['total_prize_bytes']:,}")
    print(f"total_brotli_bytes today:               {s['total_brotli_bytes']:,}")
    print(f"H2/H0 aggregate ratio:                  {s['ratio_h2_over_h0']:.4f}")
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
