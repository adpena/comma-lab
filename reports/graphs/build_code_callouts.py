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
            "Byte-layout fix",
            ROOT / "submissions" / "robust_current" / "inflate.sh",
            132,
            133,
            "The flat path forces rawvideo output to `rgb24`.",
        ),
        (
            "Explicit color contract",
            ROOT / "submissions" / "robust_current" / "compress.sh",
            79,
            90,
            "The encoded AV1 stream now carries explicit `tv/bt709` metadata.",
        ),
        (
            "Rule-faithful payload accounting",
            ROOT / "src" / "comma_lab" / "install.py",
            9,
            20,
            "The honest payload under test is explicit and small.",
        ),
        (
            "AV1 + ROI fail-fast guard",
            ROOT / "submissions" / "robust_current" / "compress.sh",
            68,
            71,
            "Unsupported AV1+ROI combinations fail loudly instead of silently drifting into x265-only behavior.",
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
