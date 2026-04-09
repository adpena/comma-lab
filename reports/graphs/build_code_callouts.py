#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "graphs" / "code_callouts.md"


def extract(path: Path, start: int, end: int) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[start - 1 : end])


def main() -> int:
    snippets = [
        (
            "Learned post-filter selector",
            ROOT / "submissions" / "robust_current" / "inflate.sh",
            176,
            181,
            "The promoted inflate path can route into the tiny learned post-filter.",
        ),
        (
            "Runtime payload includes learned assets",
            ROOT / "src" / "comma_lab" / "install.py",
            12,
            22,
            "The honest installed payload explicitly includes the post-filter script and weights.",
        ),
        (
            "Shipped post-filter module",
            ROOT / "submissions" / "robust_current" / "inflate_postfilter.py",
            1,
            40,
            "The filter is a tiny residual CNN loaded from shipped int8 weights.",
        ),
    ]

    parts = ["# code callouts", "", "Small, measured implementation details tied to the major score and rigor changes.", ""]
    for title, path, start, end, note in snippets:
        parts.extend(
            [
                f"## {title}",
                "",
                f"- file: `{path.relative_to(ROOT)}`",
                f"- why it matters: {note}",
                "",
                "```bash",
                extract(path, start, end),
                "```",
                "",
            ]
        )
    OUT.write_text("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
