#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-tensor saliency-vs-bytes heatmap.

WHEN TO USE: when you have a saliency dict from
:mod:`tac.score_gradient_param_saliency` (or a comparable Fisher-diagonal /
mean(theta^2) prior) AND a per-tensor byte map of an archive, and you want
to know WHICH tensors are over-allocated bytes relative to their score
saliency. Those are the tensors a per-tensor allocator (B6 Lagrangian-ADMM /
Path B step 5/6) should coarsen first.

WHAT IT REVEALS: for each tensor in the saliency dict, computes
saliency_per_byte = saliency_score / byte_cost, then ranks tensors. The
operator can then pick a per-tensor coarsening budget for the bottom N% by
saliency-per-byte without losing the top tensors that drive score.

Two input modes:
  - ``--saliency-json <path>`` — load saliency dict ``{name: float}``
    (canonical: ``tac.score_gradient_param_saliency.compute_score_gradient_param_saliency``)
  - ``--saliency-equal`` — fallback: use uniform 1.0 saliency across all
    tensors. This degenerates to a "byte allocation only" view but lets the
    tool run on archives where saliency hasn't been computed yet.

Byte map input:
  - ``--byte-map-json <path>`` — load ``{tensor_name: byte_count}``
  - ``--archive-byte-profile <path>`` — load
    ``tac.archive_byte_profile`` JSON output and pull tensor sizes from
    ``parser_sections``

NOT a score claim. Tagged ``[diagnostic: per-tensor saliency heatmap]``.

Output:
  experiments/results/xray_per_tensor_saliency_heatmap_<timestamp>/
    saliency_heatmap.json
    saliency_heatmap.md
    rebuild_command.txt

Usage:
  .venv/bin/python tools/xray_per_tensor_saliency_heatmap.py \
      --saliency-json reports/saliency_a1.json \
      --byte-map-json reports/per_tensor_bytes_a1.json \
      [--bottom-n-percent 25]
      [--label a1_target178000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "xray_per_tensor_saliency_heatmap_v1"
TOOL = "tools/xray_per_tensor_saliency_heatmap.py"


def load_saliency(saliency_json: Path | None, *, equal: bool, byte_map: dict[str, int]) -> dict[str, float]:
    """Load per-tensor saliency dict OR fall back to uniform.

    The saliency dict from :mod:`tac.score_gradient_param_saliency` has shape
    ``{tensor_name: float}`` where the float is the mean of squared parameter
    gradients of a score-surrogate w.r.t. that tensor.
    """
    if equal:
        return {name: 1.0 for name in byte_map}
    if saliency_json is None:
        raise ValueError("must pass either --saliency-json or --saliency-equal")
    raw = json.loads(saliency_json.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"saliency JSON must be a dict, got {type(raw).__name__}")
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    if not out:
        raise ValueError("saliency JSON yielded zero entries")
    return out


def load_byte_map(
    byte_map_json: Path | None,
    archive_byte_profile: Path | None,
) -> dict[str, int]:
    """Load per-tensor byte map dict.

    Either ``byte_map_json`` (direct ``{name: int}``) or
    ``archive_byte_profile`` (JSON output of :mod:`tac.archive_byte_profile`,
    interpreting each ``parser_sections`` entry as one tensor).
    """
    if byte_map_json is not None:
        raw = json.loads(byte_map_json.read_text())
        if not isinstance(raw, dict):
            raise ValueError("byte map JSON must be a dict")
        return {str(k): int(v) for k, v in raw.items() if int(v) > 0}
    if archive_byte_profile is not None:
        raw = json.loads(archive_byte_profile.read_text())
        # Try both ``parser_sections`` (HNeRV monolithic) and top-level
        # ``sections`` (multi-member ZIP)
        sections = raw.get("parser_sections") or raw.get("sections") or []
        out: dict[str, int] = {}
        for s in sections:
            if not isinstance(s, dict):
                continue
            name = s.get("name") or s.get("tensor_name") or s.get("section_name")
            byts = s.get("compressed_bytes") or s.get("bytes") or s.get("payload_bytes")
            if name is not None and byts is not None and int(byts) > 0:
                out[str(name)] = int(byts)
        if not out:
            raise ValueError("archive byte profile yielded zero tensor entries")
        return out
    raise ValueError("must pass --byte-map-json or --archive-byte-profile")


def build_heatmap(
    saliency: dict[str, float],
    byte_map: dict[str, int],
    *,
    bottom_n_percent: float = 25.0,
) -> dict:
    """Compute per-tensor saliency-per-byte rankings.

    For every tensor present in BOTH saliency and byte_map:
      - byte_count
      - saliency_score
      - saliency_per_byte = saliency_score / byte_count

    Output is sorted by saliency_per_byte ASCENDING (most over-allocated
    first — those are the highest-priority coarsening targets).

    The bottom ``bottom_n_percent``% by saliency-per-byte are flagged
    ``coarsen_priority=True`` to mark them for the next allocator pass.
    """
    if bottom_n_percent < 0 or bottom_n_percent > 100:
        raise ValueError(f"bottom_n_percent must be in [0,100], got {bottom_n_percent}")

    common = sorted(set(saliency.keys()) & set(byte_map.keys()))
    only_in_saliency = sorted(set(saliency.keys()) - set(byte_map.keys()))
    only_in_byte_map = sorted(set(byte_map.keys()) - set(saliency.keys()))

    rows: list[dict] = []
    for name in common:
        b = int(byte_map[name])
        s = float(saliency[name])
        spb = s / b if b > 0 else 0.0
        rows.append({
            "tensor_name": name,
            "byte_count": b,
            "saliency_score": s,
            "saliency_per_byte": spb,
        })
    rows.sort(key=lambda r: r["saliency_per_byte"])

    cutoff = max(1, int(round(len(rows) * bottom_n_percent / 100.0))) if rows else 0
    for i, row in enumerate(rows):
        row["coarsen_priority"] = i < cutoff
        row["rank"] = i

    total_bytes = sum(r["byte_count"] for r in rows)
    coarsen_bytes = sum(r["byte_count"] for r in rows if r["coarsen_priority"])
    coarsen_saliency = sum(r["saliency_score"] for r in rows if r["coarsen_priority"])
    total_saliency = sum(r["saliency_score"] for r in rows)
    return {
        "total_tensors_compared": len(common),
        "tensors_only_in_saliency": only_in_saliency,
        "tensors_only_in_byte_map": only_in_byte_map,
        "bottom_n_percent": bottom_n_percent,
        "coarsen_target_count": cutoff,
        "coarsen_target_bytes": int(coarsen_bytes),
        "coarsen_target_saliency_share": (
            coarsen_saliency / total_saliency if total_saliency > 0 else 0.0
        ),
        "total_bytes": int(total_bytes),
        "total_saliency": float(total_saliency),
        "rows": rows,
    }


def render_markdown(report: dict, regen_header: str, label: str) -> str:
    h = report["heatmap"]
    lines = [regen_header, ""]
    lines.append(f"# Per-tensor saliency heatmap ({label})")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- tensors compared: **{h['total_tensors_compared']}**")
    lines.append(f"- tensors only-in-saliency: {len(h['tensors_only_in_saliency'])}")
    lines.append(f"- tensors only-in-byte-map: {len(h['tensors_only_in_byte_map'])}")
    lines.append(f"- bottom-N%: **{h['bottom_n_percent']}%**")
    lines.append(f"- coarsen target: **{h['coarsen_target_count']} tensors**, "
                 f"**{h['coarsen_target_bytes']:,} bytes**, "
                 f"**{h['coarsen_target_saliency_share']:.4%}** of total saliency")
    lines.append(f"- total bytes: {h['total_bytes']:,}")
    lines.append("")
    lines.append("## Per-tensor (sorted ascending by saliency_per_byte = coarsen-first)")
    lines.append("")
    lines.append("| rank | tensor | bytes | saliency | sal/byte | coarsen |")
    lines.append("|---:|---|---:|---:|---:|:---:|")
    for row in h["rows"][:50]:
        flag = "**Y**" if row["coarsen_priority"] else ""
        lines.append(
            f"| {row['rank']} | `{row['tensor_name']}` | {row['byte_count']:,} | "
            f"{row['saliency_score']:.4g} | {row['saliency_per_byte']:.4g} | {flag} |"
        )
    if len(h["rows"]) > 50:
        lines.append(f"| ... | ({len(h['rows']) - 50} more) | | | | |")
    lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: per-tensor saliency heatmap]`. "
        "Tensors with **low saliency_per_byte** are over-allocated bytes "
        "relative to their contribution to the score-surrogate. The "
        "`coarsen_priority` flag identifies the bottom-N% as next allocator "
        "targets. Saliency itself is a per-tensor priority heuristic, not a "
        "score claim. Diagnostic only."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-tensor saliency-vs-bytes heatmap. Diagnostic only."
    )
    parser.add_argument("--saliency-json", type=Path, default=None)
    parser.add_argument("--saliency-equal", action="store_true",
                        help="Fallback: uniform 1.0 saliency for every tensor in byte_map")
    parser.add_argument("--byte-map-json", type=Path, default=None)
    parser.add_argument("--archive-byte-profile", type=Path, default=None)
    parser.add_argument("--bottom-n-percent", type=float, default=25.0)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        byte_map = load_byte_map(args.byte_map_json, args.archive_byte_profile)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading byte map: {e}", file=sys.stderr)
        return 2

    try:
        saliency = load_saliency(args.saliency_json, equal=args.saliency_equal, byte_map=byte_map)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading saliency: {e}", file=sys.stderr)
        return 2

    try:
        heatmap = build_heatmap(saliency, byte_map, bottom_n_percent=args.bottom_n_percent)
    except ValueError as e:
        print(f"ERROR building heatmap: {e}", file=sys.stderr)
        return 2

    if heatmap["total_tensors_compared"] == 0:
        print("ERROR: zero tensor names in common between saliency and byte_map", file=sys.stderr)
        return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"xray_per_tensor_saliency_heatmap_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    state_hash = hashlib.sha256(
        json.dumps(heatmap, sort_keys=True).encode()
    ).hexdigest()[:16]

    report = {
        "schema_version": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": state_hash,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only",
        "label": args.label,
        "saliency_source": (
            "uniform_equal" if args.saliency_equal
            else str(args.saliency_json) if args.saliency_json
            else "unknown"
        ),
        "byte_map_source": (
            str(args.byte_map_json) if args.byte_map_json
            else str(args.archive_byte_profile) if args.archive_byte_profile
            else "unknown"
        ),
        "heatmap": heatmap,
    }
    out_json = out_dir / "saliency_heatmap.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "saliency_heatmap.md"
    out_md.write_text(render_markdown(report, regen, args.label))

    cli_parts = [".venv/bin/python tools/xray_per_tensor_saliency_heatmap.py"]
    if args.saliency_json:
        cli_parts.append(f"--saliency-json {args.saliency_json}")
    if args.saliency_equal:
        cli_parts.append("--saliency-equal")
    if args.byte_map_json:
        cli_parts.append(f"--byte-map-json {args.byte_map_json}")
    if args.archive_byte_profile:
        cli_parts.append(f"--archive-byte-profile {args.archive_byte_profile}")
    cli_parts.append(f"--bottom-n-percent {args.bottom_n_percent}")
    cli_parts.append(f"--label {args.label}")
    (out_dir / "rebuild_command.txt").write_text(" \\\n  ".join(cli_parts) + "\n")

    print(f"[xray-saliency] wrote {out_json}")
    print(f"[xray-saliency] wrote {out_md}")
    print(
        f"[xray-saliency] {heatmap['total_tensors_compared']} tensors | "
        f"coarsen target = {heatmap['coarsen_target_count']} tensors / "
        f"{heatmap['coarsen_target_bytes']:,} bytes / "
        f"{heatmap['coarsen_target_saliency_share']:.4%} saliency"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
