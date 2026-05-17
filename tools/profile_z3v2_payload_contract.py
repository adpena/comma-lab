#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile Z3HV2 payload authority without claiming score movement."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.analysis.z3v2_payload_profile import (  # noqa: E402
    profile_z3v2_archive,
    render_markdown,
    write_profile_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--print-md", action="store_true")
    args = parser.parse_args()

    profile = profile_z3v2_archive(args.archive)
    write_profile_outputs(
        profile,
        json_out=args.output_json,
        markdown_out=args.output_md,
    )
    if args.print_md:
        print(render_markdown(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
