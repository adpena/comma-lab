#!/usr/bin/env python3
"""Audit whether preserved Task #602 outputs can enter the Task #603 receiver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.mdl_member_carrier_preflight import (  # noqa: E402
    MdlMemberCarrierPreflightConfigV1,
    MdlMemberCarrierPreflightProgramV1,
    write_mdl_member_carrier_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-allowed", choices=("false", "true"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.execution_allowed != "false":
            raise DirectDescriptionError("MDL member carrier preflight is read-only")
        config = MdlMemberCarrierPreflightConfigV1.model_validate_json(args.config.read_bytes())
        program = MdlMemberCarrierPreflightProgramV1(
            config_path=str(args.config),
            output_path=str(args.output),
        )
        semantic_argv = program.compile_consumer_argv()
        if tuple(sys.argv) != semantic_argv[2:]:
            raise DirectDescriptionError("consumer argv differs from typed program")
        result, output = write_mdl_member_carrier_preflight(config, args.output)
    except (OSError, ValueError, DirectDescriptionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"{result['verdict']} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
