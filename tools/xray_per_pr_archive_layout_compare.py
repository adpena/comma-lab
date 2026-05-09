#!/usr/bin/env python3
"""Side-by-side per-archive byte-section layout xray (schema-agnostic).

WHEN TO USE: when comparing 2+ archives at byte level — e.g. PR101 vs PR102
vs PR103 (the 4 hour 8 minute medal-band cluster). The binary forensics
dossier observed PR101 (gold) and PR103 (silver) shipped DIFFERENT layouts on
the SAME decoder weights; the deltas were 3-line bias corrections + sidecar
encoding choices. Without a side-by-side layout xray, those structural
differences are buried under raw byte counts.

Schema-agnostic complement to ``tools/audit_hnerv_section_candidate_diff.py``
(which is HNeRV-cluster-specific with hardcoded section names). This tool
walks the ZIP central directory of any contest archive and emits an N-way
SHARED/DIVERGED/MISSING matrix. Use the HNeRV-specific tool when you already
know you're inside the cluster; use this tool for cross-cluster comparison
or for archives whose schema has not been formalized yet.

WHAT IT REVEALS: a markdown matrix where rows are sections (union across all
archives) and columns are archives. Each cell shows
(compressed_bytes / uncompressed_bytes / sha256[:8]). Sections that exist in
some archives but not others are flagged. Sections present in all archives
with identical SHA-256 are marked "shared".

Operationally this answers: "what bytes did SajayR (PR101) change vs rem2
(PR103)?" — so the next medal-band candidate doesn't waste cycles
re-engineering shared structure.

NOT a score claim. NOT a CUDA dispatch input. Tagged
`[diagnostic: per-PR archive layout xray]` per CLAUDE.md.

Output:
  experiments/results/xray_per_pr_archive_layout_compare_<timestamp>/
    layout_compare.json     — per-section per-archive matrix
    layout_compare.md       — markdown matrix with regen header
    rebuild_command.txt

Usage:
  .venv/bin/python tools/xray_per_pr_archive_layout_compare.py \
      --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
      --archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
      --label pr101 --label pr106
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "xray_per_pr_archive_layout_compare_v1"
TOOL = "tools/xray_per_pr_archive_layout_compare.py"


def archive_layout(archive_path: Path) -> dict:
    """Return per-section layout for a single archive.

    Each section: {compressed_bytes, uncompressed_bytes, sha256, crc32}.
    Plus a per-archive sha256 of the file as a whole.
    """
    archive_path = Path(archive_path)
    file_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sections: list[dict] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            with zf.open(info, "r") as f:
                payload = f.read()
            sections.append({
                "name": info.filename,
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "crc32": int(info.CRC),
                "compress_method": int(info.compress_type),
            })
    return {
        "archive_path": str(archive_path),
        "archive_sha256": file_sha,
        "file_size_bytes": int(archive_path.stat().st_size),
        "section_count": len(sections),
        "sections": sections,
    }


def build_compare_matrix(layouts: list[dict], labels: list[str]) -> dict:
    """Compute the per-section comparison matrix across N archives.

    For each unique section name (union across all archives):
      - per-archive: compressed_bytes, uncompressed_bytes, sha256_short
      - present_in: list[label] of archives containing it
      - shared_sha256_across_all: bool (all archives have IDENTICAL bytes)
      - distinct_sha256_count: int (number of distinct content variants)
    """
    # Preserve insertion order from first archive that introduces the name
    section_order: "OrderedDict[str, None]" = OrderedDict()
    for lay in layouts:
        for s in lay["sections"]:
            section_order.setdefault(s["name"], None)

    rows: list[dict] = []
    for name in section_order:
        per_archive: dict[str, dict | None] = {}
        shas: set[str] = set()
        present_in: list[str] = []
        for lay, lab in zip(layouts, labels):
            match = next((s for s in lay["sections"] if s["name"] == name), None)
            if match is None:
                per_archive[lab] = None
            else:
                per_archive[lab] = {
                    "compressed_bytes": match["compressed_bytes"],
                    "uncompressed_bytes": match["uncompressed_bytes"],
                    "sha256_short": match["sha256"][:12],
                    "sha256": match["sha256"],
                    "compress_method": match["compress_method"],
                }
                shas.add(match["sha256"])
                present_in.append(lab)
        rows.append({
            "section_name": name,
            "per_archive": per_archive,
            "present_in": present_in,
            "missing_from": [lab for lab in labels if lab not in present_in],
            "distinct_sha256_count": len(shas),
            "shared_sha256_across_all": len(shas) == 1 and len(present_in) == len(labels),
        })

    # Aggregate stats
    total_unique_sections = len(rows)
    shared_in_all = sum(1 for r in rows if r["shared_sha256_across_all"])
    present_in_all_diff_content = sum(
        1 for r in rows
        if len(r["present_in"]) == len(labels)
        and r["distinct_sha256_count"] > 1
    )
    not_present_in_all = sum(1 for r in rows if len(r["present_in"]) < len(labels))

    return {
        "labels": labels,
        "total_unique_sections": total_unique_sections,
        "shared_sha256_across_all_archives": shared_in_all,
        "present_in_all_but_diff_content": present_in_all_diff_content,
        "not_present_in_all_archives": not_present_in_all,
        "rows": rows,
    }


def render_markdown(report: dict, regen_header: str) -> str:
    lines: list[str] = [regen_header, ""]
    lines.append("# Per-PR archive layout xray")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Archives")
    lines.append("")
    lines.append("| Label | bytes | sha256[:12] | sections |")
    lines.append("|---|---:|---|---:|")
    for lab, lay in zip(report["compare"]["labels"], report["per_archive_layouts"]):
        lines.append(
            f"| `{lab}` | {lay['file_size_bytes']:,} | "
            f"`{lay['archive_sha256'][:12]}` | "
            f"{lay['section_count']} |"
        )
    lines.append("")
    cmp = report["compare"]
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- total unique sections: **{cmp['total_unique_sections']}**")
    lines.append(f"- shared SHA across ALL archives: **{cmp['shared_sha256_across_all_archives']}**")
    lines.append(f"- present-in-all but diff content: **{cmp['present_in_all_but_diff_content']}** (the exploitable rows)")
    lines.append(f"- not present in every archive: **{cmp['not_present_in_all_archives']}**")
    lines.append("")
    labels = cmp["labels"]
    lines.append("## Per-section matrix")
    lines.append("")
    header = "| section | " + " | ".join(f"{lab} (cb/sha)" for lab in labels) + " | distinct |"
    sep = "|---|" + "|".join(["---:"] * len(labels)) + "|---:|"
    lines.append(header)
    lines.append(sep)
    for row in cmp["rows"]:
        cells: list[str] = []
        for lab in labels:
            entry = row["per_archive"].get(lab)
            if entry is None:
                cells.append("—")
            else:
                cells.append(
                    f"{entry['compressed_bytes']:,} / `{entry['sha256_short']}`"
                )
        marker = ""
        if row["shared_sha256_across_all"]:
            marker = " (SHARED)"
        elif len(row["present_in"]) < len(labels):
            marker = f" (missing in {','.join(row['missing_from'])})"
        elif row["distinct_sha256_count"] > 1:
            marker = " (DIVERGED)"
        cells_str = " | ".join(cells)
        lines.append(
            f"| `{row['section_name']}`{marker} | {cells_str} | "
            f"{row['distinct_sha256_count']} |"
        )
    lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: per-PR archive layout xray]`. SHARED rows are "
        "structural carry-over (don't re-engineer). DIVERGED rows mark exact "
        "byte-level differences across archives — these are where each PR's "
        "innovation lives. MISSING rows mark schema additions/removals."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-by-side per-archive layout xray. Diagnostic only."
    )
    parser.add_argument(
        "--archive", action="append", required=True,
        help="Archive ZIP (repeat for N-way comparison)",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="Per-archive label (repeat in same order)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir (default: experiments/results/xray_per_pr_archive_layout_compare_<ts>/)",
    )
    args = parser.parse_args(argv)

    archives = [Path(a) for a in args.archive]
    if len(archives) < 2:
        print("ERROR: need >=2 --archive for comparison", file=sys.stderr)
        return 2
    labels = args.label or [Path(a).stem for a in args.archive]
    if len(labels) != len(archives):
        print(
            f"ERROR: --label count ({len(labels)}) != --archive count ({len(archives)})",
            file=sys.stderr,
        )
        return 2

    for ar in archives:
        if not ar.exists():
            print(f"ERROR: archive not found: {ar}", file=sys.stderr)
            return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"xray_per_pr_archive_layout_compare_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    layouts = [archive_layout(ar) for ar in archives]
    cmp = build_compare_matrix(layouts, labels)
    state_hash = hashlib.sha256(
        "|".join(lay["archive_sha256"] for lay in layouts).encode()
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
        "per_archive_layouts": layouts,
        "compare": cmp,
    }

    out_json = out_dir / "layout_compare.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "layout_compare.md"
    out_md.write_text(render_markdown(report, regen))

    cli = (
        ".venv/bin/python tools/xray_per_pr_archive_layout_compare.py \\\n  "
        + " \\\n  ".join(f"--archive {a}" for a in args.archive)
    )
    if args.label:
        cli += " \\\n  " + " \\\n  ".join(f"--label {lab}" for lab in args.label)
    (out_dir / "rebuild_command.txt").write_text(cli + "\n")

    print(f"[xray-pr-layout] wrote {out_json}")
    print(f"[xray-pr-layout] wrote {out_md}")
    print(
        f"[xray-pr-layout] {len(archives)} archives | "
        f"{cmp['total_unique_sections']} unique sections | "
        f"{cmp['shared_sha256_across_all_archives']} shared | "
        f"{cmp['present_in_all_but_diff_content']} diverged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
