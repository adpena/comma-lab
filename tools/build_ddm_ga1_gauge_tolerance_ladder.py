#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the local-only DDM GA1 gauge-tolerance receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tac.optimization.ddm_ga1_gauge_tolerance_ladder import (  # noqa: E402
    canonical_json_bytes,
    compile_ga1_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    receipt = compile_ga1_receipt(config_path, repository_root=REPOSITORY_ROOT)
    output = (REPOSITORY_ROOT / json.loads(config_path.read_bytes())["output_receipt"]).resolve()
    if not output.is_relative_to(REPOSITORY_ROOT):
        parser.error("output_receipt must resolve inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(receipt) + b"\n")
    os.replace(temporary, output)
    print(output.relative_to(REPOSITORY_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
