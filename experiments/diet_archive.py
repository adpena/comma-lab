"""CLI driver for `tac.archive_diet_pack.diet_pack`.

Usage:
    .venv/bin/python experiments/diet_archive.py <input.zip> <output.zip>
    .venv/bin/python experiments/diet_archive.py <input.zip> <output.zip> --quality 11 --no-verify
    .venv/bin/python experiments/diet_archive.py <input.zip> <output.zip> --json

Score deltas printed by this tool are RATE-ONLY and tagged [advisory only].
The only authoritative measurement is contest-CUDA `inflate.sh` followed by
`upstream/evaluate.py` on the EXACT archive bytes that will be submitted.

Designed by codex gpt-5.5 xhigh under Stage 3/4 of the orchestrated workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.archive_diet_pack import diet_pack


def _format_human(stats: dict) -> str:
    lines = [
        f"[archive-diet] input  : {stats['input_bytes']:,} bytes",
        f"[archive-diet] output : {stats['output_bytes']:,} bytes",
        f"[archive-diet] saved  : {stats['savings_bytes']:,} bytes "
        f"({100.0 * stats['savings_bytes'] / max(stats['input_bytes'], 1):.2f}%)",
        f"[archive-diet] score  : -{stats['savings_score_pts']:.4f} [advisory only]",
        f"[archive-diet] verify : bit_exact={stats['bit_exact']}",
        "[archive-diet] components:",
    ]
    for name, sizes in sorted(stats["components"].items()):
        delta = sizes["in"] - sizes["out"]
        pct = 100.0 * delta / max(sizes["in"], 1)
        lines.append(
            f"  {name:32s} {sizes['in']:>10,} -> {sizes['out']:>10,}  "
            f"({delta:+,} bytes, {pct:+.2f}%)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only archive diet packer. Repacks a renderer/SegMap archive "
            "with deterministic Brotli q=11 outer compression and arithmetic-"
            "coded SegMap qint streams. Produces lossless byte savings."
        )
    )
    parser.add_argument("input_archive", type=Path, help="Input .zip archive")
    parser.add_argument("output_archive", type=Path, help="Output .zip archive")
    parser.add_argument(
        "--quality",
        type=int,
        default=11,
        help="Brotli quality 0-11 (default 11, matches Quantizr).",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip bit-exact verification of output archive (faster).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human summary.",
    )
    args = parser.parse_args(argv)

    stats = diet_pack(
        args.input_archive,
        args.output_archive,
        brotli_quality=args.quality,
        verify=args.verify,
    )

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(_format_human(stats))

    if not stats["bit_exact"]:
        print("[archive-diet] ERROR: bit_exact=False; output archive is LOSSY", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
