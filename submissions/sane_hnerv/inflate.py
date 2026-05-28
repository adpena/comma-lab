#!/usr/bin/env python
"""sane_hnerv contest-compliant inflate runtime.

Reads ``archive_dir/0.bin`` via the vendored substrate parser; writes one
contest ``.raw`` stream per ``file_list`` entry. NO scorer-network imports
(strict-scorer-rule per CLAUDE.md "Strict scorer rule" non-negotiable).

PYTHONPATH self-containment per Catalog #295: vendored substrate package
lives under ``src/tac/substrates/sane_hnerv/`` adjacent to this file; sister
``src/tac/substrates/_shared/inflate_runtime.py`` provides the canonical
``select_inflate_device`` per Catalog #205.

Wave N+45 BIND step (2026-05-28): canonical submission_dir for sane_hnerv;
sister of the trainer's emitted submission/ under experiments/results/.
Verbatim mirror of ``src/tac/substrates/sane_hnerv/inflate.py::main_cli``
(≤100 LOC per HNeRV parity L4).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from tac.substrates.sane_hnerv.inflate import (  # noqa: E402
    inflate_one_video,
)
from tac.substrates._shared.inflate_runtime import (  # noqa: E402
    raw_output_path,
    select_inflate_device,
)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    archive_bytes = (archive_dir / "0.bin").read_bytes()
    device = select_inflate_device()
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, line), device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
