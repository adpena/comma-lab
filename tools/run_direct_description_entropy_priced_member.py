#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the local-only Task #603/#613 v3 entropy or v4 structured member solve at n64."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _bootstrap_repo_python() -> None:
    candidates = tuple(root / ".venv/bin/python" for root in (REPO_ROOT, *REPO_ROOT.parents))
    interpreter = next((path for path in candidates if path.is_file()), None)
    if interpreter is None:
        print("REFUSE: repository Python environment unavailable", file=sys.stderr)
        raise SystemExit(2)
    try:
        already_pinned = Path(sys.executable).resolve() == interpreter.resolve()
    except OSError:
        already_pinned = False
    if already_pinned:
        return
    if os.environ.get("DDM_ENTROPY_PRICED_MEMBER_REPO_PYTHON_BOOTSTRAPPED") == "1":
        print("REFUSE: repository Python bootstrap did not converge", file=sys.stderr)
        raise SystemExit(2)
    environment = dict(os.environ)
    environment["DDM_ENTROPY_PRICED_MEMBER_REPO_PYTHON_BOOTSTRAPPED"] = "1"
    try:
        os.execve(
            str(interpreter),
            [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )
    except OSError as exc:
        print(f"REFUSE: repository Python bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    _bootstrap_repo_python()

from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    DirectDescriptionEntropyPricedMemberConfigV1,
    DirectDescriptionEntropyPricedMemberProgramV1,
    DirectDescriptionStratumStructuredMemberConfigV1,
    DirectDescriptionStratumStructuredMemberProgramV1,
    run_entropy_priced_member_n64,
    run_stratum_structured_member_n64,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402

Config = DirectDescriptionEntropyPricedMemberConfigV1 | DirectDescriptionStratumStructuredMemberConfigV1


def _read_config(path: Path) -> Config:
    try:
        payload = path.read_bytes()
        schema = json.loads(payload).get("schema")
        if schema == "DirectDescriptionEntropyPricedMemberConfigV1":
            return DirectDescriptionEntropyPricedMemberConfigV1.model_validate_json(payload)
        if schema == "DirectDescriptionStratumStructuredMemberConfigV1":
            return DirectDescriptionStratumStructuredMemberConfigV1.model_validate_json(payload)
        raise DirectDescriptionError(f"entropy-priced member config schema is unknown: {schema!r}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"entropy-priced member typed config is unreadable: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-allowed", choices=("false", "true"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.execution_allowed != "false":
            raise DirectDescriptionError("entropy-priced member solve only compiles --execution-allowed false")
        config = _read_config(args.config)
        if isinstance(config, DirectDescriptionStratumStructuredMemberConfigV1):
            structured_program = DirectDescriptionStratumStructuredMemberProgramV1(
                config_path=str(args.config),
                output_directory=str(args.output_dir),
            )
            receipt, receipt_path = run_stratum_structured_member_n64(
                config,
                output_directory=args.output_dir,
                semantic_argv=structured_program.compile_consumer_argv(),
            )
        else:
            entropy_program = DirectDescriptionEntropyPricedMemberProgramV1(
                config_path=str(args.config),
                output_directory=str(args.output_dir),
            )
            receipt, receipt_path = run_entropy_priced_member_n64(
                config,
                output_directory=args.output_dir,
                semantic_argv=entropy_program.compile_consumer_argv(),
            )
    except (DirectDescriptionError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt_path": str(receipt_path), "receipt": receipt}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
