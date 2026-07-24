#!/usr/bin/env python3
"""Measure the exact n600 solve-as-oracle residual correction price."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.ddm_is1_oracle_diff_exact_residual import (  # noqa: E402
    OracleDiffPriceConfigV1,
    run_exact_oracle_diff_price,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = OracleDiffPriceConfigV1.model_validate_json(
        config_path.read_bytes()
    )
    receipt = run_exact_oracle_diff_price(
        config,
        args.output_root,
        resume=args.resume,
        tool_path=Path(__file__).resolve(),
        config_path=config_path,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

