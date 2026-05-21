#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the parser-visible procedural replacement surface matrix.

This is a static planner, not an evaluator. It ranks where seed-derived
procedural replacement can plausibly apply after the FEC6 parser-safe subset
closed at zero. All outputs are non-promotional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.procedural_replacement_surfaces import build_surface_matrix_payload  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments/results" / f"procedural_replacement_surface_matrix_{stamp}"


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Procedural Replacement Surface Matrix",
        "",
        f"**Axis:** `{payload['axis_tag']}`",
        f"**Evidence grade:** `{payload['evidence_grade']}`",
        "**Authority:** `score_claim=false`, `promotion_eligible=false`, "
        "`ready_for_exact_eval_dispatch=false`",
        "",
        "| Rank | Substrate | Surface | Status | Parser-visible | Adapter | Bytes saved | Predicted dS | Blocker |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(payload["surfaces"], start=1):
        lines.append(
            "| {idx} | `{substrate}` | `{surface}` | `{status}` | {visible} | "
            "{adapter} | {saved:,} | {delta:.9f} | {blocker} |".format(
                idx=idx,
                substrate=row["substrate_id"],
                surface=row["surface_id"],
                status=row["candidate_status"],
                visible="yes" if row["parser_visible"] else "no",
                adapter="yes" if row["requires_archive_adapter"] else "no",
                saved=int(row["predicted_bytes_saved"]),
                delta=float(row["predicted_delta_s"]),
                blocker=str(row["blocker"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `parser_visible=true` means the section is visible to the substrate parser;",
            "  it does not mean the scorer is insensitive to the bytes.",
            "- `raw_byte_mutation_parse_safe=false` blocks null-byte mutation inside",
            "  compressed streams, but can still allow whole-section replacement when",
            "  the archive grammar has a procedural-aware adapter.",
            "- Predicted dS is the rate-axis-only canonical equation #26 arithmetic;",
            "  it is not a contest score claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for surface_matrix.{json,md}; default uses UTC timestamp.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    payload = build_surface_matrix_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "surface_matrix.json"
    md_path = output_dir / "surface_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(payload, md_path)
    print(f"Wrote {_display_path(json_path)}")
    print(f"Wrote {_display_path(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
