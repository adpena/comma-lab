#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run local-only Task #603 full-resolution DDM measurement rungs 1-3."""

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
    if os.environ.get("DDM_MEASUREMENT_LADDER_REPO_PYTHON_BOOTSTRAPPED") == "1":
        print("REFUSE: repository Python bootstrap did not converge", file=sys.stderr)
        raise SystemExit(2)
    environment = dict(os.environ)
    environment["DDM_MEASUREMENT_LADDER_REPO_PYTHON_BOOTSTRAPPED"] = "1"
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

from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    DirectDescriptionMeasurementLadderConfigV1,
    DirectDescriptionMeasurementLadderProgramV1,
    run_measurement_ladder,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402


def _read_config(path: Path) -> DirectDescriptionMeasurementLadderConfigV1:
    try:
        return DirectDescriptionMeasurementLadderConfigV1.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise DirectDescriptionError(f"measurement-ladder typed config is unreadable: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-allowed", choices=("false", "true"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.execution_allowed != "false":
            raise DirectDescriptionError("measurement ladder only compiles --execution-allowed false")
        config = _read_config(args.config)
        program = DirectDescriptionMeasurementLadderProgramV1(
            config_path=str(args.config), output_directory=str(args.output_dir)
        )
        receipt, receipt_path = run_measurement_ladder(
            config,
            output_directory=args.output_dir,
            semantic_argv=program.compile_consumer_argv(),
        )
    except (DirectDescriptionError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"receipt_path": str(receipt_path), "receipt": receipt}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
