#!/usr/bin/env python3
"""PR101 theoretical-floor analyzer — task #397 (engineering hygiene).

Reports the canonical Shannon-floor stack for the PR101 substrate, per
the council's encoder-saturation finding (memo
``feedback_grand_council_three_falsifications_path_forward_20260507.md``).

For each tensor in PR101's state_dict, computes:
  - Uniform Shannon floor: log2(N_QUANT) bits/element
  - Empirical Shannon floor: per-tensor empirical entropy
  - Joint floor lower bound: weighted-avg empirical * n_elements
  - Brotli-Optuna empirical: 178,144 B (canonical reference)
  - All current empirical anchors from cathedral_autopilot_evidence.jsonl

Output: a markdown table that gives the operator a one-page view of
the encoder-side Shannon ladder + the empirical anchors, so the
encoder-vs-architecture decision is unambiguous.

CLAUDE.md compliance: pure CPU + numpy + math; no scorer load; no
contest score claims; output tagged ``[CPU-prep theoretical-floor]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_theoretical_floor_analyzer.py"
SCHEMA_VERSION = "pr101_theoretical_floor_analyzer.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
BROTLI_OPTUNA_REFERENCE_BYTES = 178_144


def shannon_entropy_bits(symbols: np.ndarray) -> tuple[float, int]:
    """Per-element Shannon entropy + n_unique_symbols."""
    if symbols.size == 0:
        return 0.0, 0
    counts = Counter(int(s) for s in symbols.flatten().tolist())
    total = sum(counts.values())
    h = 0.0
    for n in counts.values():
        if n == 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h, len(counts)


def analyze(state_dict_path: Path, evidence_jsonl: Path | None) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    uniform_bits = math.log2(N_QUANT)
    rows: list[dict] = []
    n_total = 0
    weighted_h_sum = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        h_bits, n_unique = shannon_entropy_bits(qt.q_i8)
        rows.append({
            "name": name,
            "n_elements": int(qt.q_i8.size),
            "uniform_bits_per_element": uniform_bits,
            "empirical_bits_per_element": h_bits,
            "n_unique_symbols": n_unique,
            "entropy_ratio": h_bits / uniform_bits if uniform_bits > 0 else 1.0,
        })
        n_total += int(qt.q_i8.size)
        weighted_h_sum += h_bits * int(qt.q_i8.size)

    weighted_h = weighted_h_sum / max(n_total, 1)
    iid_floor_bytes = int(round(weighted_h * n_total / 8.0))
    archive_iid_floor = iid_floor_bytes + ARCHIVE_OVERHEAD_BYTES
    uniform_total_bytes = int(round(uniform_bits * n_total / 8.0))
    archive_uniform_floor = uniform_total_bytes + ARCHIVE_OVERHEAD_BYTES

    # Load empirical anchors from the canonical evidence feed
    empirical_anchors: list[dict] = []
    if evidence_jsonl and evidence_jsonl.is_file():
        for line in evidence_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    empirical_anchors.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Group by technique, take the latest empirical_archive_bytes per technique
    by_technique: dict[str, dict] = {}
    for row in empirical_anchors:
        name = row.get("technique")
        if name and "empirical_archive_bytes" in row:
            by_technique[name] = row

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[CPU-prep theoretical-floor]",
        "score_claim": False,
        "input_state_dict": str(state_dict_path),
        "evidence_jsonl": str(evidence_jsonl) if evidence_jsonl else None,
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "n_total_elements": n_total,
        "n_quant": N_QUANT,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "uniform_bits_per_element": uniform_bits,
        "weighted_empirical_bits_per_element": weighted_h,
        "uniform_floor_bytes": archive_uniform_floor,
        "iid_empirical_floor_bytes": archive_iid_floor,
        "brotli_optuna_reference_bytes": BROTLI_OPTUNA_REFERENCE_BYTES,
        "brotli_iid_excess_bytes": BROTLI_OPTUNA_REFERENCE_BYTES - archive_iid_floor,
        "brotli_iid_excess_pct": (
            100.0 * (BROTLI_OPTUNA_REFERENCE_BYTES - archive_iid_floor) / archive_iid_floor
            if archive_iid_floor > 0 else 0.0
        ),
        "empirical_anchors_from_evidence_jsonl": list(by_technique.values()),
        "per_tensor": rows,
    }


def render_summary(manifest: dict) -> str:
    lines: list[str] = []
    lines.append("=== PR101 Theoretical Floor Analysis ===")
    lines.append("")
    lines.append(f"Substrate: PR101 HNeRV decoder, {manifest['n_tensors']} tensors, "
                 f"{manifest['n_total_elements']:,} elements")
    lines.append(f"Quantization: INT8 symmetric N_QUANT={manifest['n_quant']}")
    lines.append("")
    lines.append("ENCODER-SIDE LADDER (Shannon floors + empirical):")
    lines.append("")
    lines.append(f"  Uniform Shannon floor:           {manifest['uniform_floor_bytes']:>10,} B")
    lines.append(f"  IID empirical Shannon floor:     {manifest['iid_empirical_floor_bytes']:>10,} B")
    lines.append(f"  Brotli + Optuna (current):       {manifest['brotli_optuna_reference_bytes']:>10,} B")
    excess_pct = manifest['brotli_iid_excess_pct']
    lines.append(f"  Brotli excess vs IID floor:      {manifest['brotli_iid_excess_bytes']:>10,} B "
                 f"({excess_pct:+.2f}%)")
    lines.append("")
    lines.append("EMPIRICAL ANCHORS (from cathedral_autopilot_evidence.jsonl):")
    lines.append("")
    if manifest.get("empirical_anchors_from_evidence_jsonl"):
        for row in sorted(
            manifest["empirical_anchors_from_evidence_jsonl"],
            key=lambda r: r.get("empirical_archive_bytes", 0),
        ):
            tech = row.get("technique", "?")
            bytes_ = row.get("empirical_archive_bytes", 0)
            verdict = row.get("contest_dispatch_verdict", "")
            verdict_str = f"  [verdict: {verdict}]" if verdict else ""
            lines.append(f"  {tech:<42s}  {bytes_:>10,} B{verdict_str}")
    else:
        lines.append("  (no evidence rows found)")
    lines.append("")
    lines.append("INTERPRETATION (per grand council 2026-05-07):")
    lines.append("")
    if excess_pct < 5.0:
        lines.append(f"  Brotli sits {excess_pct:.1f}% above IID Shannon floor — encoder lane SATURATED.")
        lines.append(f"  Architecture lane (sparsity, arch_shrink) has 5-10x more headroom.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--evidence-jsonl", type=Path,
                   default=REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--summary-text", action="store_true")
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = analyze(args.state_dict, args.evidence_jsonl)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_theoretical_floor_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.summary_text:
        print(render_summary(manifest))
    else:
        print(render_summary(manifest))
        print(f"\n(manifest: {args.output_json})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
